from __future__ import annotations

import json
import hashlib
import os
import pty
import select
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from pydantic import BaseModel, Field

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from checks.common import existing_file, run_sandbox_python, sandbox_path

MAX_EPOCHS = 200
DEFAULT_EPOCHS = 10
DEFAULT_TRAIN_LIMIT = 2048
FIXED_TEST_LIMIT = 512
REMOTE_TIMEOUT_SECONDS = 3600
REMOTE_INFRASTRUCTURE_ERROR_MARKERS = (
    "No space left on device",
    "Disk quota exceeded",
    "Errno 28",
    "ENOSPC",
)


class Input(BaseModel):
    model_path: str = Field(
        description="Sandbox-relative path to a Python PyTorch model definition or YOLO .pt checkpoint. For COCO Python models, the check trains a segmentation mask proxy and reports mAP50-95. For SpeechOcean Python models, the check trains IPA character sequence outputs and reports phonetic accuracy rate (1 - character error rate). For YOLO checkpoints, the score is the trained checkpoint's mAP only; pretrained validation before training is reported for context but does not count."
    )
    data_path: str = Field(
        description="Sandbox-relative path to the dataset. Use data.zip for COCO or reference_dataset for SpeechOcean."
    )
    epochs: int = Field(
        default=DEFAULT_EPOCHS,
        ge=1,
        le=MAX_EPOCHS,
        description="Number of short training epochs to run.",
    )
    train_limit: int = Field(
        default=DEFAULT_TRAIN_LIMIT,
        ge=8,
        le=100000,
        description="Maximum number of training examples to use for this check.",
    )


def remote_training_enabled() -> bool:
    return bool(os.environ.get("REMOTE_TRAIN_HOST"))


def is_remote_infrastructure_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        marker.lower() in lowered for marker in REMOTE_INFRASTRUCTURE_ERROR_MARKERS
    )


def run_password_command(
    args: list[str],
    *,
    password: str | None,
    timeout_seconds: int = REMOTE_TIMEOUT_SECONDS,
) -> tuple[int, str]:
    started = time.monotonic()
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execvp(args[0], args)
    output = bytearray()
    password_sent = False
    try:
        while True:
            if time.monotonic() - started > timeout_seconds:
                os.kill(pid, 9)
                raise subprocess.TimeoutExpired(args, timeout_seconds)
            ready, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in ready:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""
                if chunk:
                    output.extend(chunk)
                    lower = bytes(output[-4096:]).lower()
                    if (
                        password
                        and not password_sent
                        and (b"password:" in lower or b"passphrase" in lower)
                    ):
                        os.write(master_fd, password.encode() + b"\n")
                        password_sent = True
                    if b"are you sure you want to continue connecting" in lower:
                        os.write(master_fd, b"yes\n")
                else:
                    break
            finished_pid, status = os.waitpid(pid, os.WNOHANG)
            if finished_pid:
                try:
                    while True:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        output.extend(chunk)
                except OSError:
                    pass
                if os.WIFEXITED(status):
                    return os.WEXITSTATUS(status), output.decode(errors="replace")
                if os.WIFSIGNALED(status):
                    return 128 + os.WTERMSIG(status), output.decode(errors="replace")
                return 1, output.decode(errors="replace")
    finally:
        os.close(master_fd)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status), output.decode(errors="replace")
    if os.WIFSIGNALED(status):
        return 128 + os.WTERMSIG(status), output.decode(errors="replace")
    return 1, output.decode(errors="replace")


def ssh_args(host: str, remote_command: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        host,
        remote_command,
    ]


def remote_exec(
    host: str,
    remote_command: str,
    *,
    password: str | None,
    timeout_seconds: int = REMOTE_TIMEOUT_SECONDS,
) -> str:
    code, output = run_password_command(
        ssh_args(host, remote_command),
        password=password,
        timeout_seconds=timeout_seconds,
    )
    if code != 0:
        raise RuntimeError(
            output.strip() or f"remote command failed with exit code {code}"
        )
    return output.strip()


def scp_to_remote(
    source: Path,
    host: str,
    destination: str,
    *,
    password: str | None,
    timeout_seconds: int = REMOTE_TIMEOUT_SECONDS,
) -> None:
    args = [
        "scp",
        "-r",
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(source),
        f"{host}:{destination}",
    ]
    code, output = run_password_command(
        args, password=password, timeout_seconds=timeout_seconds
    )
    if code != 0:
        raise RuntimeError(output.strip() or f"scp failed with exit code {code}")


def parse_training_metrics(detail: str) -> dict:
    for line in reversed(detail.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"training script did not return JSON metrics:\n{detail}")


def parse_gpu_stats(detail: str) -> dict[str, float | int | None]:
    peak_memory_mb = 0
    peak_utilization_percent = 0
    peak_power_watts = 0.0
    samples = 0
    for line in detail.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            memory_mb = int(float(parts[0]))
            utilization_percent = int(float(parts[1]))
            power_watts = float(parts[2])
        except ValueError:
            continue
        peak_memory_mb = max(peak_memory_mb, memory_mb)
        peak_utilization_percent = max(peak_utilization_percent, utilization_percent)
        peak_power_watts = max(peak_power_watts, power_watts)
        samples += 1
    return {
        "gpu_peak_memory_mb": peak_memory_mb if samples else None,
        "gpu_peak_utilization_percent": peak_utilization_percent if samples else None,
        "gpu_peak_power_watts": peak_power_watts if samples else None,
        "gpu_telemetry_samples": samples,
    }


def prior_best_train_score(state: RunState) -> float | None:
    best: float | None = None
    for results in state.results_by_iteration:
        result = results.get("train.py") if isinstance(results, dict) else None
        if result is None:
            continue
        score = result.score
        if best is None or score > best:
            best = score
    return best


def result_from_metrics(payload: dict) -> CheckResult:
    final_score = float(payload.get("final_accuracy", payload.get("final_score", 0.0)))
    initial_score = float(
        payload.get("initial_accuracy", payload.get("initial_score", 0.0))
    )
    improved = bool(payload.get("improved", final_score > initial_score))
    metric = payload.get("metric", "accuracy")
    feedback = (
        f"backend={payload.get('backend', 'local')} "
        f"gpu={payload.get('gpu', 'none')} "
        f"dataset={payload.get('dataset')} "
        f"train_examples={payload.get('train_examples')} "
        f"test_examples={payload.get('test_examples')} "
        f"epochs={payload.get('epochs')} "
        f"metric={metric} "
        f"initial_score={initial_score:.4f} "
        f"final_score={final_score:.4f}"
        + (
            f" torch_device={payload['torch_device']}"
            if payload.get("torch_device") is not None
            else ""
        )
        + (
            f" gpu_peak_memory_mb={int(payload['gpu_peak_memory_mb'])}"
            if payload.get("gpu_peak_memory_mb") is not None
            else ""
        )
        + (
            f" gpu_peak_utilization_percent={int(payload['gpu_peak_utilization_percent'])}"
            if payload.get("gpu_peak_utilization_percent") is not None
            else ""
        )
        + (
            f" gpu_peak_power_watts={float(payload['gpu_peak_power_watts']):.1f}"
            if payload.get("gpu_peak_power_watts") is not None
            else ""
        )
        + (
            f" gpu_telemetry_samples={int(payload['gpu_telemetry_samples'])}"
            if payload.get("gpu_telemetry_samples") is not None
            else ""
        )
        + (
            f" trained_checkpoint_accuracy={float(payload['trained_checkpoint_accuracy']):.4f}"
            if "trained_checkpoint_accuracy" in payload
            else ""
        )
        + (
            f" pretrained_checkpoint_accuracy={float(payload['pretrained_checkpoint_accuracy']):.4f}"
            if "pretrained_checkpoint_accuracy" in payload
            else ""
        )
    )
    prior_best = payload.get("prior_best_score")
    if prior_best is not None:
        prior_best = float(prior_best)
        feedback += f" prior_best_score={prior_best:.4f}"
        if final_score <= prior_best + 1e-9:
            feedback += (
                " recommendation=score_did_not_improve; if all checks already pass, "
                f"try increasing the epochs/train_limit (epochs can go up to {MAX_EPOCHS})."
                " Keep the last fully passing artifact as a fallback. Only replace it "
                "with a materially different candidate after making the smallest change "
                "needed for the failing check; for STM32 activation-memory failures, "
                "reduce sequence length/frame count and intermediate channels before "
                "trying larger models."
            )
        else:
            feedback += (
                " recommendation=score_improved; preserve this artifact as the "
                "current fallback and only replace it with candidates that also pass "
                "all deployability checks."
            )
    if payload.get("pretrained_checkpoint_accuracy") is not None:
        threshold = float(os.environ.get("REMOTE_TRAIN_MAP_THRESHOLD", "0.15"))
        if final_score < threshold:
            feedback += (
                f" threshold={threshold:.4f} "
                "trained_checkpoint_below_threshold; pretrained_checkpoint_accuracy "
                "is diagnostic only and does not count as the trained score"
            )
    return CheckResult(
        success=improved,
        score=final_score,
        score_unit=str(payload.get("score_unit", "accuracy")),
        feedback=feedback,
    )


def dataset_cache_key(data_path: Path) -> str:
    if data_path.is_file():
        stat = data_path.stat()
        return f"{data_path.stem}-{stat.st_size}-{file_sha256(data_path)[:12]}"
    return data_path.name


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_signature(path: Path) -> tuple[int, int]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return len(files), sum(item.stat().st_size for item in files)


def remote_file_matches(
    *,
    host: str,
    remote_path: str,
    local_path: Path,
    password: str | None,
) -> bool:
    expected_size = local_path.stat().st_size
    expected_sha = file_sha256(local_path)
    command = (
        f"if [ ! -f {shlex.quote(remote_path)} ]; then echo missing; exit 0; fi; "
        "python3 - "
        f"{shlex.quote(remote_path)} <<'PY'\n"
        "import hashlib\n"
        "import pathlib\n"
        "import sys\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "digest = hashlib.sha256()\n"
        "with path.open('rb') as handle:\n"
        "    for chunk in iter(lambda: handle.read(1024 * 1024), b''):\n"
        "        digest.update(chunk)\n"
        "print(path.stat().st_size, digest.hexdigest())\n"
        "PY"
    )
    output = remote_exec(host, command, password=password)
    last_line = output.splitlines()[-1].strip() if output else ""
    if last_line == "missing":
        return False
    parts = last_line.split()
    return (
        len(parts) == 2 and parts[0] == str(expected_size) and parts[1] == expected_sha
    )


def remote_directory_matches(
    *,
    host: str,
    remote_path: str,
    local_path: Path,
    password: str | None,
) -> bool:
    expected_count, expected_size = directory_signature(local_path)
    command = (
        f"if [ ! -d {shlex.quote(remote_path)} ]; then echo missing; exit 0; fi; "
        "python3 - "
        f"{shlex.quote(remote_path)} <<'PY'\n"
        "import pathlib\n"
        "import sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "files = [path for path in root.rglob('*') if path.is_file()]\n"
        "print(len(files), sum(path.stat().st_size for path in files))\n"
        "PY"
    )
    output = remote_exec(host, command, password=password)
    last_line = output.splitlines()[-1].strip() if output else ""
    if last_line == "missing":
        return False
    parts = last_line.split()
    return (
        len(parts) == 2
        and parts[0] == str(expected_count)
        and parts[1] == str(expected_size)
    )


def upload_file_atomically(
    *,
    source: Path,
    host: str,
    destination: str,
    password: str | None,
) -> None:
    parent = str(Path(destination).parent)
    temp_destination = f"{destination}.tmp-{uuid.uuid4().hex}"
    remote_exec(host, f"mkdir -p {shlex.quote(parent)}", password=password)
    try:
        scp_to_remote(source, host, shlex.quote(temp_destination), password=password)
        remote_exec(
            host,
            f"mv {shlex.quote(temp_destination)} {shlex.quote(destination)}",
            password=password,
        )
    except Exception:
        try:
            remote_exec(
                host, f"rm -f {shlex.quote(temp_destination)}", password=password
            )
        except Exception:
            pass
        raise


def upload_directory_atomically(
    *,
    source: Path,
    host: str,
    destination: str,
    password: str | None,
) -> None:
    parent = str(Path(destination).parent)
    temp_destination = f"{destination}.tmp-{uuid.uuid4().hex}"
    remote_exec(
        host,
        f"mkdir -p {shlex.quote(parent)} && rm -rf {shlex.quote(temp_destination)}",
        password=password,
    )
    try:
        scp_to_remote(source, host, shlex.quote(temp_destination), password=password)
        remote_exec(
            host,
            f"rm -rf {shlex.quote(destination)} && mv {shlex.quote(temp_destination)} {shlex.quote(destination)}",
            password=password,
        )
    except Exception:
        try:
            remote_exec(
                host, f"rm -rf {shlex.quote(temp_destination)}", password=password
            )
        except Exception:
            pass
        raise


def remote_cache_target(input_data_path: str, data_path: Path, cache_root: str) -> str:
    key = dataset_cache_key(data_path)
    if input_data_path == "data.zip" or data_path.name == "data.zip":
        return f"{cache_root.rstrip('/')}/datasets/{key}.zip"
    if input_data_path == "reference_dataset" or data_path.name == "reference_dataset":
        return f"{cache_root.rstrip('/')}/datasets/{key}"
    return ""


def score_unit_for_dataset(input_data_path: str, data_path: Path) -> str:
    if input_data_path == "data.zip" or data_path.suffix.lower() == ".zip":
        return "mAP50-95"
    if input_data_path == "reference_dataset" or data_path.is_dir():
        return "phonetic_accuracy_rate"
    return "accuracy"


def choose_remote_gpu_command() -> str:
    return r"""python3 - <<'PY'
import subprocess
rows = subprocess.check_output([
    "nvidia-smi",
    "--query-gpu=index,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
], text=True).strip().splitlines()
best = None
for row in rows:
    index, mem, util = [part.strip() for part in row.split(",")]
    item = (int(mem), int(util), int(index))
    if best is None or item < best:
        best = item
print(best[2])
PY"""


def setup_remote_python(
    *,
    host: str,
    password: str | None,
    cache_root: str,
    remote_python: str,
) -> None:
    command = (
        f"mkdir -p {shlex.quote(cache_root)} && "
        f"if [ ! -x {shlex.quote(remote_python)} ]; then "
        f"python3 -m venv {shlex.quote(str(Path(remote_python).parent.parent))} && "
        f"{shlex.quote(remote_python)} -m pip install -U pip wheel; "
        f"fi && "
        f"{shlex.quote(remote_python)} - <<'PY'\n"
        f"import importlib.util, subprocess, sys\n"
        f"def import_ok(name):\n"
        f"    try:\n"
        f"        __import__(name); return True\n"
        f"    except Exception:\n"
        f"        return False\n"
        f"if not import_ok('torch') or not import_ok('torchvision'):\n"
        f"    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--index-url', 'https://download.pytorch.org/whl/cu128', '--force-reinstall', 'torch', 'torchvision'])\n"
        f"missing = [p for p, m in [('numpy','numpy'),('pillow','PIL'),('pyarrow','pyarrow'),('pyyaml','yaml'),('ultralytics','ultralytics')] if importlib.util.find_spec(m) is None]\n"
        f"if missing:\n"
        f"    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])\n"
        f"PY"
    )
    remote_exec(host, command, password=password, timeout_seconds=1800)


def yolo_runner_source(input: Input) -> str:
    source = r"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import zipfile

import yaml

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("ULTRALYTICS_SETTINGS", "/tmp/ultralytics-settings.json")

from ultralytics import YOLO

MODEL_PATH = pathlib.Path(__MODEL_PATH__).resolve()
DATA_PATH = pathlib.Path(__DATA_PATH__).resolve()
EPOCHS = __EPOCHS__
TRAIN_LIMIT = __TRAIN_LIMIT__
TEST_LIMIT = __TEST_LIMIT__
IMGSZ = __IMGSZ__
BATCH = __BATCH__


def extract_coco_zip(path: pathlib.Path) -> pathlib.Path:
    work = pathlib.Path("dataset").resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(work)
    coco = work / "coco"
    if not coco.exists():
        raise SystemExit("Expected COCO zip to contain a coco/ directory")
    return coco


def limited_list(coco: pathlib.Path, source_name: str, output_name: str, limit: int) -> str:
    source = coco / source_name
    if not source.exists():
        raise SystemExit(f"Missing {source_name} in COCO dataset")
    lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if limit > 0:
        lines = lines[:limit]
    output = coco / output_name
    output.write_text("\n".join(lines) + "\n")
    return output.name


def make_data_yaml(coco: pathlib.Path) -> pathlib.Path:
    original = yaml.safe_load((coco / "data.yaml").read_text())
    names = original.get("names")
    data = {
        "path": str(coco),
        "train": limited_list(coco, "train.txt", "train_subset.txt", TRAIN_LIMIT),
        "val": limited_list(coco, "val.txt", "val_subset.txt", TEST_LIMIT),
        "test": limited_list(coco, "test.txt", "test_subset.txt", TEST_LIMIT) if (coco / "test.txt").exists() else limited_list(coco, "val.txt", "test_subset.txt", TEST_LIMIT),
        "names": names,
    }
    output = coco / "edgedl_data.yaml"
    output.write_text(yaml.safe_dump(data, sort_keys=False))
    return output


def metric_map(results):
    for attr in ("seg", "box"):
        value = getattr(results, attr, None)
        if value is not None and getattr(value, "map", None) is not None:
            return float(value.map)
    if getattr(results, "results_dict", None):
        for key, value in results.results_dict.items():
            if "map50-95" in key.lower() or "map" in key.lower():
                try:
                    return float(value)
                except Exception:
                    pass
    raise RuntimeError("Could not read mAP50-95 from Ultralytics validation results")


coco = extract_coco_zip(DATA_PATH)
data_yaml = make_data_yaml(coco)

model = YOLO(str(MODEL_PATH))
initial = model.val(data=str(data_yaml), imgsz=IMGSZ, batch=BATCH, device=0, task="segment", workers=8, verbose=False)
pretrained_map = metric_map(initial)

train_result = model.train(
    data=str(data_yaml),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,
    task="segment",
    workers=8,
    project="runs",
    name="train",
    exist_ok=True,
    plots=False,
    verbose=False,
)
best_path = pathlib.Path(train_result.save_dir) / "weights" / "best.pt"
trained = YOLO(str(best_path if best_path.exists() else MODEL_PATH))
final = trained.val(data=str(data_yaml), imgsz=IMGSZ, batch=BATCH, device=0, task="segment", workers=8, verbose=False)
final_map = metric_map(final)
threshold = float(os.environ.get("REMOTE_TRAIN_MAP_THRESHOLD", "0.15"))

print(json.dumps({
    "dataset": "coco",
    "train_examples": min(TRAIN_LIMIT, len((coco / "train_subset.txt").read_text().splitlines())),
    "test_examples": min(TEST_LIMIT, len((coco / "val_subset.txt").read_text().splitlines())),
    "epochs": EPOCHS,
    "initial_accuracy": 0.0,
    "final_accuracy": final_map,
    "trained_checkpoint_accuracy": final_map,
    "pretrained_checkpoint_accuracy": pretrained_map,
    "improved": final_map >= threshold,
    "metric": "mAP50-95",
    "score_unit": "mAP50-95",
    "best_model_path": str(best_path),
}))
"""
    return (
        source.replace("__MODEL_PATH__", repr(input.model_path))
        .replace("__DATA_PATH__", repr(input.data_path))
        .replace("__EPOCHS__", str(input.epochs))
        .replace("__TRAIN_LIMIT__", str(input.train_limit))
        .replace("__TEST_LIMIT__", str(FIXED_TEST_LIMIT))
        .replace("__IMGSZ__", os.environ.get("REMOTE_TRAIN_YOLO_IMGSZ", "640"))
        .replace("__BATCH__", os.environ.get("REMOTE_TRAIN_YOLO_BATCH", "64"))
    )


def run_remote_training(
    *,
    state: RunState,
    input: Input,
    model_path: Path,
    data_path: Path,
    runner_source: str,
) -> CheckResult:
    failure_score_unit = score_unit_for_dataset(input.data_path, data_path)
    host = os.environ["REMOTE_TRAIN_HOST"]
    password = os.environ.get("REMOTE_TRAIN_PASSWORD")
    remote_root = os.environ.get("REMOTE_TRAIN_ROOT")
    remote_cache = os.environ.get("REMOTE_TRAIN_CACHE")
    if not remote_root or not remote_cache:
        raise ExperimentSetupError(
            "REMOTE_TRAIN_ROOT and REMOTE_TRAIN_CACHE must be set when REMOTE_TRAIN_HOST is set."
        )
    remote_python = os.environ.get(
        "REMOTE_TRAIN_PYTHON", f"{remote_cache.rstrip('/')}/venv/bin/python"
    )
    if shutil.which("ssh") is None or shutil.which("scp") is None:
        raise ExperimentSetupError("ssh and scp are required for remote training.")

    run_id = (
        f"train-{state.trial_index}-{state.iteration_index}-{uuid.uuid4().hex[:10]}"
    )
    remote_run_dir = f"{remote_root.rstrip('/')}/{run_id}"
    remote_data_path = remote_cache_target(input.data_path, data_path, remote_cache)
    if not remote_data_path:
        remote_data_path = f"{remote_run_dir}/{data_path.name}"

    try:
        try:
            setup_remote_python(
                host=host,
                password=password,
                cache_root=remote_cache,
                remote_python=remote_python,
            )
        except RuntimeError as exc:
            raise ExperimentSetupError(
                f"remote training Python setup failed: {exc}"
            ) from exc
        remote_exec(
            host,
            f"mkdir -p {shlex.quote(remote_run_dir)} {shlex.quote(remote_cache)}",
            password=password,
        )

        is_yolo_run = (
            model_path.suffix.lower() == ".pt" and data_path.suffix.lower() == ".zip"
        )
        active_runner_source = (
            yolo_runner_source(input) if is_yolo_run else runner_source
        )

        with tempfile.TemporaryDirectory(prefix="edgedl-remote-train-") as tmp:
            bundle_name = f"{run_id}.bundle"
            bundle = Path(tmp) / bundle_name
            remote_model_rel = Path(input.model_path)
            if remote_model_rel.is_absolute() or ".." in remote_model_rel.parts:
                remote_model_rel = Path("candidate_model.py")
            remote_model_path = f"{remote_run_dir}/{remote_model_rel.as_posix()}"
            remote_runner_source = active_runner_source.replace(
                repr(input.model_path), repr(remote_model_path)
            ).replace(repr(input.data_path), repr(remote_data_path))
            local_model_copy = bundle / remote_model_rel
            local_model_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_path, local_model_copy)
            (bundle / "runner.py").write_text(remote_runner_source)
            scp_to_remote(
                bundle,
                host,
                shlex.quote(str(Path(remote_run_dir).parent)),
                password=password,
            )
        remote_exec(
            host,
            f"rm -rf {shlex.quote(remote_run_dir)} && mv {shlex.quote(str(Path(remote_run_dir).parent / bundle_name))} {shlex.quote(remote_run_dir)}",
            password=password,
        )

        if remote_cache_target(input.data_path, data_path, remote_cache):
            if data_path.is_file():
                if not remote_file_matches(
                    host=host,
                    remote_path=remote_data_path,
                    local_path=data_path,
                    password=password,
                ):
                    upload_file_atomically(
                        source=data_path,
                        host=host,
                        destination=remote_data_path,
                        password=password,
                    )
            else:
                if not remote_directory_matches(
                    host=host,
                    remote_path=remote_data_path,
                    local_path=data_path,
                    password=password,
                ):
                    upload_directory_atomically(
                        source=data_path,
                        host=host,
                        destination=remote_data_path,
                        password=password,
                    )
        else:
            if data_path.is_file():
                upload_file_atomically(
                    source=data_path,
                    host=host,
                    destination=remote_data_path,
                    password=password,
                )
            else:
                upload_directory_atomically(
                    source=data_path,
                    host=host,
                    destination=remote_data_path,
                    password=password,
                )

        gpu = (
            remote_exec(host, choose_remote_gpu_command(), password=password)
            .splitlines()[-1]
            .strip()
        )
        stats_path = f"{remote_run_dir}/gpu_stats.csv"
        status_path = f"{remote_run_dir}/status.txt"
        remote_command = (
            f"cd {shlex.quote(remote_run_dir)} || exit 1; "
            f"(while true; do "
            f"nvidia-smi --id={shlex.quote(gpu)} "
            f"--query-gpu=memory.used,utilization.gpu,power.draw "
            f"--format=csv,noheader,nounits; "
            f"sleep 1; "
            f"done >> {shlex.quote(stats_path)} 2>/dev/null) & monitor_pid=$!; "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu)} "
            f"{shlex.quote(remote_python)} runner.py "
            f"> stdout.txt 2> stderr.txt; "
            f"status=$?; echo $status > {shlex.quote(status_path)}; "
            f"kill $monitor_pid 2>/dev/null || true; "
            f"wait $monitor_pid 2>/dev/null || true; "
            f"exit 0"
        )
        remote_exec(
            host,
            remote_command,
            password=password,
            timeout_seconds=REMOTE_TIMEOUT_SECONDS,
        )
        metrics_text = remote_exec(
            host,
            f"cat {shlex.quote(remote_run_dir)}/stdout.txt {shlex.quote(remote_run_dir)}/stderr.txt",
            password=password,
        )
        gpu_stats_text = remote_exec(
            host,
            f"test -f {shlex.quote(stats_path)} && cat {shlex.quote(stats_path)} || true",
            password=password,
        )
        status_text = (
            remote_exec(
                host,
                f"test -f {shlex.quote(status_path)} && cat {shlex.quote(status_path)} || echo missing",
                password=password,
            )
            .splitlines()[-1]
            .strip()
        )
        if status_text != "0":
            raise RuntimeError(
                "remote runner failed"
                + (
                    f" with exit code {status_text}"
                    if status_text != "missing"
                    else " before writing an exit status"
                )
                + ":\n"
                + (metrics_text.strip() or "(remote stdout/stderr were empty)")
            )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"remote training timed out after {exc.timeout}s") from exc
    except ExperimentSetupError:
        raise
    except RuntimeError as exc:
        if is_remote_infrastructure_error(str(exc)):
            raise ExperimentSetupError(
                "Remote training infrastructure failed: "
                f"{exc}. Free space or fix the remote host, then rerun/resume."
            ) from exc
        return CheckResult(
            success=False,
            score=0.0,
            score_unit=failure_score_unit,
            feedback=f"remote training failed: {exc}",
        )
    finally:
        if os.environ.get("REMOTE_TRAIN_KEEP_RUNS") != "1":
            try:
                remote_exec(
                    host,
                    f"rm -rf {shlex.quote(remote_run_dir)} {shlex.quote(remote_run_dir)}.bundle",
                    password=password,
                    timeout_seconds=300,
                )
            except Exception:
                pass

    try:
        payload = parse_training_metrics(metrics_text)
    except ValueError as exc:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit=failure_score_unit,
            feedback=str(exc),
        )
    payload["backend"] = "remote"
    payload["gpu"] = gpu
    payload.update(parse_gpu_stats(gpu_stats_text))
    payload["prior_best_score"] = prior_best_train_score(state)
    return result_from_metrics(payload)


def check(state: RunState, input: Input) -> CheckResult:
    """Trains the submitted PyTorch architecture briefly and scores held-out accuracy."""
    model_path, error = existing_file(state, input.model_path)
    if error:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="accuracy",
            feedback=error,
        )
    is_yolo_checkpoint = input.model_path.endswith(".pt") and input.data_path.endswith(
        ".zip"
    )
    is_ai8x_checkpoint = input.model_path.endswith(
        ".pth.tar"
    ) or input.model_path.endswith(".pth")
    if (
        not input.model_path.endswith(".py")
        and not is_yolo_checkpoint
        and not is_ai8x_checkpoint
    ):
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="accuracy",
            feedback="train.py requires a Python PyTorch model definition, a YOLO .pt checkpoint with a COCO .zip dataset, or an ai8x .pth.tar checkpoint.",
        )
    assert model_path is not None

    try:
        data_path = sandbox_path(state, input.data_path)
    except Exception as exc:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="accuracy",
            feedback=str(exc),
        )
    if not data_path.exists():
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="accuracy",
            feedback=f"{input.data_path} does not exist",
        )

    smoke = """
from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import pathlib
import random
import sys
import wave
import zipfile
import collections

import numpy as np
os.environ.setdefault("HOME", "/tmp")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor")
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

MODEL_PATH = pathlib.Path(__MODEL_PATH__).resolve()
DATA_PATH = pathlib.Path(__DATA_PATH__).resolve()
EPOCHS = __EPOCHS__
TRAIN_LIMIT = __TRAIN_LIMIT__
TEST_LIMIT = __TEST_LIMIT__
BATCH_SIZE = 8
IS_AI8X_CHECKPOINT = MODEL_PATH.name.endswith(".pth.tar") or MODEL_PATH.suffix.lower() in {".pth", ".tar"}
IMAGE_SIZE = 32 if IS_AI8X_CHECKPOINT else 64
SPEECH_SAMPLES = 16000

torch.manual_seed(7)
np.random.seed(7)
random.seed(7)
torch.set_num_threads(1)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AI8XConvBlock(nn.Module):
    def __init__(self, op, pool=False):
        super().__init__()
        self.op = op
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return F.relu(self.op(x))


class AI85NASCifarNetProxy(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1_1 = AI8XConvBlock(nn.Conv2d(3, 64, 3, padding=1))
        self.conv1_2 = AI8XConvBlock(nn.Conv2d(64, 32, 1))
        self.conv1_3 = AI8XConvBlock(nn.Conv2d(32, 64, 3, padding=1))
        self.conv2_1 = AI8XConvBlock(nn.Conv2d(64, 32, 3, padding=1), pool=True)
        self.conv2_2 = AI8XConvBlock(nn.Conv2d(32, 64, 1))
        self.conv3_1 = AI8XConvBlock(nn.Conv2d(64, 128, 3, padding=1), pool=True)
        self.conv3_2 = AI8XConvBlock(nn.Conv2d(128, 128, 1))
        self.conv4_1 = AI8XConvBlock(nn.Conv2d(128, 64, 3, padding=1), pool=True)
        self.conv4_2 = AI8XConvBlock(nn.Conv2d(64, 128, 3, padding=1))
        self.conv5_1 = AI8XConvBlock(nn.Conv2d(128, 128, 1), pool=True)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        for name in (
            "conv1_1",
            "conv1_2",
            "conv1_3",
            "conv2_1",
            "conv2_2",
            "conv3_1",
            "conv3_2",
            "conv4_1",
            "conv4_2",
            "conv5_1",
        ):
            x = getattr(self, name)(x)
        return self.fc(x.flatten(1))


def load_ai8x_checkpoint_model():
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise SystemExit("ai8x checkpoint must be a dictionary")
    arch = checkpoint.get("arch")
    if arch != "ai85nascifarnet":
        raise SystemExit(
            f"Unsupported ai8x checkpoint architecture {arch!r}. "
            "The train check currently supports ai85nascifarnet checkpoints."
        )
    return AI85NASCifarNetProxy(num_classes=10)


def load_model():
    if IS_AI8X_CHECKPOINT:
        return load_ai8x_checkpoint_model()

    spec = importlib.util.spec_from_file_location("candidate_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["candidate_model"] = module
    spec.loader.exec_module(module)

    factory = None
    for name in ("build_model", "create_model", "get_model", "Model", "Net"):
        obj = getattr(module, name, None)
        if obj is not None:
            factory = obj
            break
    if factory is None:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, torch.nn.Module) and obj is not torch.nn.Module and obj.__module__ == module.__name__:
                factory = obj
                break
    if factory is None:
        raise SystemExit("Expected a no-argument PyTorch nn.Module class or build_model/create_model/get_model factory")
    model = factory()
    if not hasattr(model, "parameters"):
        raise SystemExit("Model factory did not return a PyTorch module-like object")
    params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
    if not params:
        raise SystemExit("Model has no trainable parameters")
    return model


def first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def logits_from_output(output):
    tensor = first_tensor(output)
    if tensor is None:
        raise RuntimeError("forward did not return a tensor-like output")
    if tensor.ndim == 0:
        raise RuntimeError("forward returned a scalar, expected batched logits")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim > 2:
        tensor = tensor.flatten(1)
    if tensor.shape[0] == 0 or tensor.shape[1] < 2:
        raise RuntimeError(f"forward produced logits with unusable shape {tuple(tensor.shape)}")
    return tensor.float()


def batches(items, batch_size):
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        xs = torch.stack([item[0] for item in chunk])
        if torch.is_tensor(chunk[0][1]):
            try:
                ys = torch.stack([item[1] for item in chunk])
            except RuntimeError:
                ys = [item[1] for item in chunk]
        else:
            ys = torch.tensor([item[1] for item in chunk], dtype=torch.long)
        if torch.is_tensor(ys):
            ys = ys.to(DEVICE, non_blocking=True)
        else:
            ys = [item.to(DEVICE, non_blocking=True) for item in ys]
        yield xs.to(DEVICE, non_blocking=True), ys


def image_to_tensor(raw):
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def mask_for_coco_image(zip_file, image_name):
    label_name = image_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    try:
        text = zip_file.read(label_name).decode("utf-8", errors="replace")
    except KeyError:
        return None
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(mask)
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        if len(coords) >= 6 and len(coords) % 2 == 0:
            points = [
                (
                    max(0.0, min(1.0, coords[i])) * (IMAGE_SIZE - 1),
                    max(0.0, min(1.0, coords[i + 1])) * (IMAGE_SIZE - 1),
                )
                for i in range(0, len(coords), 2)
            ]
            draw.polygon(points, fill=1)
        elif len(coords) >= 4:
            cx, cy, width, height = coords[:4]
            x0 = max(0.0, cx - width / 2) * (IMAGE_SIZE - 1)
            y0 = max(0.0, cy - height / 2) * (IMAGE_SIZE - 1)
            x1 = min(1.0, cx + width / 2) * (IMAGE_SIZE - 1)
            y1 = min(1.0, cy + height / 2) * (IMAGE_SIZE - 1)
            draw.rectangle((x0, y0, x1, y1), fill=1)
    arr = np.asarray(mask, dtype=np.float32)
    if arr.sum() <= 0:
        return None
    return torch.from_numpy(arr).unsqueeze(0)


def load_coco_from_zip(path):
    with zipfile.ZipFile(path) as zf:
        names = [
            name
            for name in zf.namelist()
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
            and not name.startswith("__MACOSX/")
            and "/._" not in name
        ]
        raw = {}
        for split, split_tokens in {
            "train": ("/train",),
            "test": ("/val", "/test"),
        }.items():
            raw_items = []
            for name in sorted(
                name for name in names if any(token in name for token in split_tokens)
            ):
                raw_items.append((name, None))
            raw[split] = raw_items

        def materialize(split, limit):
            items = []
            for name, _ in raw[split]:
                mask = mask_for_coco_image(zf, name)
                if mask is None:
                    continue
                try:
                    items.append((image_to_tensor(zf.read(name)), mask))
                except Exception:
                    continue
                if limit > 0 and len(items) >= limit:
                    break
            return items

        train = materialize("train", TRAIN_LIMIT)
        test = materialize("test", TEST_LIMIT)
    if not train or not test:
        raise SystemExit("COCO dataset must contain train images and val/test images with YOLO label files")
    return "coco", train, test


def audio_to_tensor(value):
    raw = None
    if isinstance(value, dict):
        raw = value.get("bytes")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    if raw is None:
        return torch.zeros(SPEECH_SAMPLES, dtype=torch.float32)
    raw = bytes(raw)
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            frames = wav.readframes(wav.getnframes())
            width = wav.getsampwidth()
            if width == 1:
                arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                arr = (arr - 128.0) / 128.0
            elif width == 2:
                arr = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
            elif width == 4:
                arr = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
            else:
                arr = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0
            channels = max(1, wav.getnchannels())
            if channels > 1 and arr.size >= channels:
                arr = arr.reshape(-1, channels).mean(axis=1)
    except Exception:
        arr = np.frombuffer(raw[:SPEECH_SAMPLES], dtype=np.uint8).astype(np.float32)
        arr = (arr - 127.5) / 127.5
    if arr.size < SPEECH_SAMPLES:
        arr = np.pad(arr, (0, SPEECH_SAMPLES - arr.size))
    else:
        arr = arr[:SPEECH_SAMPLES]
    return torch.from_numpy(arr.astype(np.float32))


def ipa_text(value):
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return " ".join(text.split())


def load_speech_ocean(path):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(f"pyarrow is required to read SpeechOcean parquet files: {exc}")

    train_file = next((path / "data").glob("train-*.parquet"), None)
    test_file = next((path / "data").glob("test-*.parquet"), None)
    if train_file is None or test_file is None:
        raise SystemExit("SpeechOcean dataset must contain data/train-*.parquet and data/test-*.parquet")

    raw_train = pq.read_table(train_file, columns=["audio", "ipa"]).slice(0, TRAIN_LIMIT).to_pylist()
    raw_test = pq.read_table(test_file, columns=["audio", "ipa"]).slice(0, TEST_LIMIT).to_pylist()
    texts = [ipa_text(row.get("ipa")) for row in raw_train + raw_test]
    chars = sorted({char for text in texts for char in text})
    if not chars:
        raise SystemExit("Need non-empty IPA targets to measure phonetic accuracy")
    global BLANK_INDEX, CHAR_TO_ID, ID_TO_CHAR
    BLANK_INDEX = 0
    CHAR_TO_ID = {char: idx + 1 for idx, char in enumerate(chars)}
    ID_TO_CHAR = {idx: char for char, idx in CHAR_TO_ID.items()}

    def convert(rows):
        converted = []
        for row in rows:
            target = torch.tensor(
                [CHAR_TO_ID[char] for char in ipa_text(row.get("ipa"))],
                dtype=torch.long,
            )
            if target.numel() > 0:
                converted.append((audio_to_tensor(row.get("audio")), target))
        return converted

    return "speechocean", convert(raw_train), convert(raw_test)


def load_dataset(path):
    if path.is_file() and path.suffix.lower() == ".zip":
        return load_coco_from_zip(path)
    if path.is_dir() and (path / "data").exists():
        return load_speech_ocean(path)
    raise SystemExit("Unsupported dataset path. Expected COCO .zip or SpeechOcean snapshot directory.")


def choose_input_transform(model, kind, sample):
    if kind == "coco":
        candidates = [
            lambda x: x,
            lambda x: F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False),
            lambda x: x.flatten(1),
        ]
    else:
        candidates = [
            lambda x: x,
            lambda x: x.unsqueeze(1),
            lambda x: x[:, :1024],
            lambda x: x.unsqueeze(1)[:, :, :1024],
        ]
    errors = []
    model.eval()
    for transform in candidates:
        try:
            x = transform(sample)
            if kind == "speechocean":
                sequence_logits_from_output(model(x), x.shape[0], max(ID_TO_CHAR) + 1)
            else:
                logits_from_output(model(x))
            return transform
        except Exception as exc:
            errors.append(str(exc))
    raise SystemExit("Model could not run on the dataset input tensors: " + "; ".join(errors[:4]))


def segmentation_logits(output, target):
    tensor = first_tensor(output)
    if tensor is None:
        raise RuntimeError("forward did not return a tensor-like output")
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise RuntimeError(f"segmentation output must be BxCxHxW, got {tuple(tensor.shape)}")
    if tensor.shape[1] != 1:
        tensor = tensor[:, :1]
    if tensor.shape[-2:] != target.shape[-2:]:
        tensor = F.interpolate(tensor, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return tensor.float()


def mask_map50_95(model, items, transform):
    thresholds = [0.50 + 0.05 * index for index in range(10)]
    hits = [0 for _ in thresholds]
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in batches(items, BATCH_SIZE):
            x = transform(x)
            logits = segmentation_logits(model(x), y)
            pred = torch.sigmoid(logits) > 0.5
            target = y > 0.5
            intersection = (pred & target).flatten(1).sum(dim=1).float()
            union = (pred | target).flatten(1).sum(dim=1).float().clamp_min(1.0)
            ious = intersection / union
            for index, threshold in enumerate(thresholds):
                hits[index] += int((ious >= threshold).sum().item())
            total += int(y.shape[0])
    if total == 0:
        raise RuntimeError("No test examples were available")
    return sum(hit / total for hit in hits) / len(thresholds)


def sequence_logits_from_output(output, batch_size, min_classes):
    tensor = first_tensor(output)
    if tensor is None:
        raise RuntimeError("forward did not return a tensor-like output")
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(1)
    elif tensor.ndim == 3:
        if tensor.shape[0] == batch_size and tensor.shape[-1] >= min_classes:
            pass
        elif tensor.shape[0] == batch_size and tensor.shape[1] >= min_classes:
            tensor = tensor.transpose(1, 2)
        elif tensor.shape[1] == batch_size and tensor.shape[-1] >= min_classes:
            tensor = tensor.transpose(0, 1)
        else:
            raise RuntimeError(f"speech output shape is not BxTxC, BxCxT, or TxBxC: {tuple(tensor.shape)}")
    elif tensor.ndim == 4:
        if tensor.shape[0] != batch_size:
            raise RuntimeError(f"speech output batch mismatch: {tuple(tensor.shape)}")
        if tensor.shape[1] >= min_classes:
            tensor = tensor.flatten(2).transpose(1, 2)
        elif tensor.shape[-1] >= min_classes:
            tensor = tensor.flatten(1, 2)
        else:
            raise RuntimeError(f"speech output does not expose at least {min_classes} classes: {tuple(tensor.shape)}")
    else:
        raise RuntimeError(f"unsupported speech output rank {tensor.ndim}: {tuple(tensor.shape)}")
    if tensor.shape[-1] < min_classes:
        raise RuntimeError(
            f"model output has {tensor.shape[-1]} classes but IPA vocabulary requires {min_classes}"
        )
    return tensor.float()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            ))
        prev = cur
    return prev[-1]


def decode_prediction(logits):
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    chars = []
    previous = None
    for idx in ids:
        if idx != BLANK_INDEX and idx != previous:
            chars.append(ID_TO_CHAR.get(idx, ""))
        previous = idx
    return "".join(chars)


def decode_target(target):
    return "".join(ID_TO_CHAR.get(int(idx), "") for idx in target.detach().cpu().tolist())


def phonetic_accuracy_rate(model, items, transform):
    min_classes = max(ID_TO_CHAR) + 1
    edits = 0
    chars = 0
    model.eval()
    with torch.no_grad():
        for x, y in batches(items, BATCH_SIZE):
            x = transform(x)
            logits = sequence_logits_from_output(model(x), x.shape[0], min_classes)
            for index, target in enumerate(y):
                truth = decode_target(target)
                pred = decode_prediction(logits[index])
                edits += edit_distance(pred, truth)
                chars += max(1, len(truth))
    if chars == 0:
        raise RuntimeError("No IPA target characters were available")
    return max(0.0, 1.0 - edits / chars)


def evaluate(model, items, transform, kind):
    if kind == "coco":
        return mask_map50_95(model, items, transform)
    if kind == "speechocean":
        return phonetic_accuracy_rate(model, items, transform)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in batches(items, BATCH_SIZE):
            x = transform(x)
            logits = logits_from_output(model(x))
            if int(y.max().item()) >= logits.shape[1]:
                raise RuntimeError(
                    f"model output has {logits.shape[1]} classes but dataset labels require {int(y.max().item()) + 1}"
                )
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    if total == 0:
        raise RuntimeError("No test examples were available")
    return correct / total


def train(model, items, transform, epochs, kind):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
        weight_decay=1e-4,
    )
    best_accuracy = None
    for _ in range(epochs):
        model.train()
        random.shuffle(items)
        for x, y in batches(items, BATCH_SIZE):
            x = transform(x)
            if kind == "coco":
                logits = segmentation_logits(model(x), y)
                loss = F.binary_cross_entropy_with_logits(logits, y.float())
            elif kind == "speechocean":
                min_classes = max(ID_TO_CHAR) + 1
                logits = sequence_logits_from_output(model(x), x.shape[0], min_classes)
                log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
                if torch.is_tensor(y):
                    targets = y.reshape(-1)
                    target_lengths = torch.full(
                        (y.shape[0],),
                        y.shape[1],
                        dtype=torch.long,
                        device=DEVICE,
                    )
                else:
                    targets = torch.cat(y)
                    target_lengths = torch.tensor(
                        [len(target) for target in y],
                        dtype=torch.long,
                        device=DEVICE,
                    )
                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=DEVICE,
                )
                loss = F.ctc_loss(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=BLANK_INDEX,
                    zero_infinity=True,
                )
            else:
                logits = logits_from_output(model(x))
                if int(y.max().item()) >= logits.shape[1]:
                    raise RuntimeError(
                        f"model output has {logits.shape[1]} classes but dataset labels require {int(y.max().item()) + 1}"
                    )
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss became non-finite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        candidate_accuracy = evaluate(model, test_items, transform, kind)
        if best_accuracy is None or candidate_accuracy > best_accuracy:
            best_accuracy = candidate_accuracy
    return best_accuracy


kind, train_items, test_items = load_dataset(DATA_PATH)
model = load_model().to(DEVICE)
sample = torch.stack([item[0] for item in train_items[:min(BATCH_SIZE, len(train_items))]]).to(DEVICE)
transform = choose_input_transform(model, kind, sample)

initial_accuracy = evaluate(model, test_items, transform, kind)
final_accuracy = train(model, train_items, transform, EPOCHS, kind)
if final_accuracy is None:
    final_accuracy = evaluate(model, test_items, transform, kind)

print(json.dumps({
    "dataset": kind,
    "train_examples": len(train_items),
    "test_examples": len(test_items),
    "epochs": EPOCHS,
    "initial_accuracy": initial_accuracy,
    "final_accuracy": final_accuracy,
    "improved": final_accuracy > initial_accuracy,
    "metric": "mAP50-95" if kind == "coco" else "phonetic_accuracy_rate",
    "score_unit": "mAP50-95" if kind == "coco" else "phonetic_accuracy_rate",
    "torch_device": str(DEVICE),
}))
"""
    smoke = (
        smoke.replace("__MODEL_PATH__", repr(input.model_path))
        .replace("__DATA_PATH__", repr(input.data_path))
        .replace("__EPOCHS__", str(input.epochs))
        .replace("__TRAIN_LIMIT__", str(input.train_limit))
        .replace("__TEST_LIMIT__", str(FIXED_TEST_LIMIT))
    )
    if remote_training_enabled():
        return run_remote_training(
            state=state,
            input=input,
            model_path=model_path,
            data_path=data_path,
            runner_source=smoke,
        )

    if is_yolo_checkpoint:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="mAP50-95",
            feedback="YOLO mAP training requires the remote GPU backend. Set REMOTE_TRAIN_HOST.",
        )

    ok, detail = run_sandbox_python(state, smoke, timeout_seconds=900)
    if not ok:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit=score_unit_for_dataset(input.data_path, data_path),
            feedback=detail,
        )

    try:
        payload = parse_training_metrics(detail)
    except ValueError as exc:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit=score_unit_for_dataset(input.data_path, data_path),
            feedback=str(exc),
        )

    payload["backend"] = "local"
    payload["prior_best_score"] = prior_best_train_score(state)
    return result_from_metrics(payload)
