import os
import shutil
import textwrap
from pathlib import Path

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel, Field
from checks.common import (
    compile_python_if_needed,
    existing_file,
    load_yaml_file,
    run_host_command,
    run_sandbox_python,
    sandbox_path,
    weighted_result,
)


class Input(BaseModel):
    model_path: str
    yaml_path: str
    sample_input_path: str | None = None
    prefix: str = Field(default="candidate", pattern=r"^[A-Za-z0-9_.-]+$")


def ensure_arm_toolchain_path() -> None:
    arm_bin = os.environ.get("ARM_GNU_TOOLCHAIN_BIN")
    if not arm_bin:
        return
    current_path = os.environ.get("PATH", "")
    paths = current_path.split(os.pathsep) if current_path else []
    if arm_bin not in paths:
        os.environ["PATH"] = arm_bin + (
            os.pathsep + current_path if current_path else ""
        )


def checkpoint_from_python_model(
    state: RunState,
    *,
    model_path: str,
    yaml_data: dict,
    prefix: str,
) -> tuple[Path | None, str | None]:
    arch = yaml_data.get("arch") if isinstance(yaml_data, dict) else None
    if not isinstance(arch, str) or not arch:
        return None, "YAML must contain an arch string to create an ai8x checkpoint"
    output_relative = f".generated/{prefix}.pth.tar"
    script = textwrap.dedent(f"""
        import importlib.util
        import inspect
        import pathlib
        import sys

        import torch

        torch.manual_seed(78000)

        model_path = pathlib.Path({model_path!r}).resolve()
        output_path = pathlib.Path({output_relative!r}).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        spec = importlib.util.spec_from_file_location("candidate_model", model_path)
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
        if not hasattr(model, "state_dict"):
            raise SystemExit("Model factory did not return a PyTorch module")

        randomized_state_dict = {{}}
        for name, tensor in model.state_dict().items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                randomized_state_dict[name] = tensor
                continue
            if tensor.numel() == 0:
                randomized_state_dict[name] = tensor.clone()
                continue
            if name.endswith(".weight") and tensor.ndim >= 2:
                values = torch.randint(
                    0,
                    2,
                    tensor.shape,
                    dtype=torch.int8,
                    device=tensor.device,
                ).to(dtype=tensor.dtype)
                randomized_state_dict[name] = values.mul(2).sub(1)
            elif name.endswith(".bias"):
                values = torch.randint(
                    1,
                    4,
                    tensor.shape,
                    dtype=torch.int8,
                    device=tensor.device,
                ).to(dtype=tensor.dtype)
                signs = torch.randint(
                    0,
                    2,
                    tensor.shape,
                    dtype=torch.int8,
                    device=tensor.device,
                ).to(dtype=tensor.dtype).mul(2).sub(1)
                randomized_state_dict[name] = values.mul(signs)
            else:
                randomized_state_dict[name] = torch.randn_like(tensor).clamp(-1, 1)

        torch.save(
            {{"arch": {arch!r}, "epoch": 0, "state_dict": randomized_state_dict}},
            output_path,
        )
        print(output_path)
        """)
    ok, detail = run_sandbox_python(state, script, timeout_seconds=120)
    if not ok:
        return None, detail
    try:
        return sandbox_path(state, output_relative), None
    except Exception as exc:
        return None, str(exc)


def check(state: RunState, input: Input) -> CheckResult:
    """Checks that the model can be synthesized to run on the MAX78000"""
    ensure_arm_toolchain_path()
    checks = []
    compression_task = (
        any(check.name == "train.py" for check in state.config.checks)
        or "compression" in state.config.task.name.lower()
    )
    seeded_reference_paths = (
        input.model_path.startswith("synthesis_examples/")
        or input.yaml_path.startswith("synthesis_examples/")
        or (
            input.sample_input_path is not None
            and input.sample_input_path.startswith("synthesis_examples/")
        )
    )
    if compression_task and seeded_reference_paths:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="fraction",
            feedback=(
                "Compression experiments cannot submit files under synthesis_examples/. "
                "Those files are seeded references for inspection only. Create a new "
                "compressed candidate model artifact, matching ai8x YAML, and sample "
                "input under a candidate-specific path."
            ),
        )

    model_path, model_error = existing_file(state, input.model_path)
    checks.append(
        ("model file exists", model_error is None, 0.25, model_error or str(model_path))
    )
    if model_error is None:
        compile_ok, compile_detail = compile_python_if_needed(state, input.model_path)
        checks.append(
            ("model file compiles when applicable", compile_ok, 0.15, compile_detail)
        )
    else:
        checks.append(
            ("model file compiles when applicable", False, 0.15, "model file missing")
        )

    yaml_path, yaml_error = existing_file(state, input.yaml_path)
    checks.append(
        ("MAX78000 YAML exists", yaml_error is None, 0.25, yaml_error or str(yaml_path))
    )
    yaml_data = None
    if yaml_error is None and yaml_path:
        yaml_data, parse_error = load_yaml_file(yaml_path)
        checks.append(
            (
                "MAX78000 YAML parses",
                parse_error is None,
                0.20,
                parse_error or "valid YAML",
            )
        )
        architecture_like = isinstance(yaml_data, dict) and any(
            key in yaml_data
            for key in ("arch", "dataset", "layers", "network", "model")
        )
        checks.append(
            (
                "YAML contains architecture fields",
                architecture_like,
                0.15,
                (
                    "found architecture-like keys"
                    if architecture_like
                    else "expected one of arch/dataset/layers/network/model"
                ),
            )
        )
    else:
        checks.append(("MAX78000 YAML parses", False, 0.20, "YAML file missing"))
        checks.append(
            ("YAML contains architecture fields", False, 0.15, "YAML file missing")
        )

    setup_ok = all(passed for _, passed, _, _ in checks)
    if not setup_ok:
        return weighted_result(checks=checks, success_threshold=1.0)

    ai8x_synthesis_dir = os.environ.get("AI8X_SYNTHESIS_DIR")
    if not ai8x_synthesis_dir:
        raise ExperimentSetupError(
            "AI8X_SYNTHESIS_DIR is not set. Run scripts/setup_max78000.sh, then source .env or export AI8X_SYNTHESIS_DIR."
        )

    ai8xize = Path(ai8x_synthesis_dir) / "ai8xize.py"
    if not ai8xize.exists():
        raise ExperimentSetupError(
            f"AI8X_SYNTHESIS_DIR does not point to ai8x-synthesis: {ai8xize} does not exist."
        )

    sdk_examples = Path(ai8x_synthesis_dir) / os.environ.get(
        "AI8X_TEST_DIR", "sdk/Examples/MAX78000/CNN"
    )
    if not sdk_examples.exists():
        raise ExperimentSetupError(
            f"MAX78000 SDK examples directory is missing: {sdk_examples}. Run scripts/setup_max78000.sh and install/download the MSDK into .data/toolchains/max78000/ai8x-synthesis/sdk."
        )

    sample_args = []
    if input.sample_input_path:
        sample_path, sample_error = existing_file(state, input.sample_input_path)
        checks.append(
            (
                "sample input exists",
                sample_error is None,
                0.05,
                sample_error or str(sample_path),
            )
        )
        if sample_error is None:
            sample_args = ["--sample-input", str(sample_path)]

    checkpoint_path = model_path
    if input.model_path.endswith(".py"):
        generated_path, generated_error = checkpoint_from_python_model(
            state,
            model_path=input.model_path,
            yaml_data=yaml_data,  # type: ignore
            prefix=input.prefix,
        )
        checks.append(
            (
                "generated ai8x checkpoint from Python model",
                generated_error is None,
                0.10,
                generated_error or str(generated_path),
            )
        )
        if generated_error is not None:
            return weighted_result(checks=checks, success_threshold=1.0)
        checkpoint_path = generated_path

    command = [
        os.environ.get("AI8X_PYTHON", "python"),
        str(ai8xize),
        "--verbose",
        "--log",
        "--test-dir",
        os.environ.get("AI8X_TEST_DIR", "sdk/Examples/MAX78000/CNN"),
        "--prefix",
        input.prefix,
        "--checkpoint-file",
        str(checkpoint_path),
        "--config-file",
        str(yaml_path),
        "--device",
        "MAX78000",
        "--compact-data",
        "--mexpress",
        "--timer",
        "0",
        "--yamllint",
        "none",
        "--display-checkpoint",
        "--fifo",
        "--mlator",
        "--overwrite",
        "--no-version-check",
        *sample_args,
    ]
    synth_ok, synth_output = run_host_command(
        command,
        cwd=Path(ai8x_synthesis_dir),
        timeout_seconds=int(os.environ.get("AI8XIZE_TIMEOUT_SECONDS", "600")),
    )
    checks.append(
        (
            "ai8xize synthesis",
            synth_ok,
            0.30,
            synth_output[-6000:] if synth_output else "ai8xize completed",
        )
    )

    project_dir = (
        Path(ai8x_synthesis_dir)
        / os.environ.get("AI8X_TEST_DIR", "sdk/Examples/MAX78000/CNN")
        / input.prefix
    )
    makefile = project_dir / "Makefile"
    if synth_ok and makefile.exists():
        if shutil.which("arm-none-eabi-gcc") is None:
            raise ExperimentSetupError(
                "arm-none-eabi-gcc is not on PATH. Run scripts/setup_max78000.sh, then source .env."
            )
        build_ok, build_output = run_host_command(
            ["make"],
            cwd=project_dir,
            timeout_seconds=int(
                os.environ.get("MAX78000_BUILD_TIMEOUT_SECONDS", "600")
            ),
        )
        checks.append(
            (
                "generated MAX78000 C builds",
                build_ok,
                0.20,
                build_output[-6000:] if build_output else "make completed",
            )
        )
    else:
        checks.append(
            (
                "generated MAX78000 C builds",
                False,
                0.20,
                (
                    f"No generated Makefile at {makefile}"
                    if synth_ok
                    else "synthesis failed"
                ),
            )
        )

    result = weighted_result(checks=checks, success_threshold=1.0)
    if result.success:
        result.feedback += (  # type: ignore
            "\nrecommendation=synthesis_passed; preserve this exact model/YAML/sample "
            "triplet as the deployable fallback. Improve mAP by adjusting training "
            "settings or making small local architecture changes, but revert to this "
            "artifact if a new candidate does not also pass synthesis."
        )
    else:
        result.feedback += (  # type: ignore
            "\nrecommendation=synthesis_failed; prioritize deployability before mAP. "
            "Start from the smallest ai8x-compatible conv-only model and YAML that "
            "passes synthesis before increasing capacity. A known-good baseline is "
            "exactly: ConvBlock modules exposing .conv2d, channels 3->16->16->8, "
            "layer 1/2 kernel_size 3x3 pad 1 ReLU, layer 3 kernel_size 1x1 pad 0 "
            "activate None, processors 0x0000000000000007 then "
            "0x000000000000ffff then 0x000000000000ffff, output_processors "
            "0x000000000000ffff then 0x000000000000ffff then "
            "0x00000000000000ff, out_offset 0x2000 then 0 then 0x2000, "
            "and data_format HWC on the first layer. Use an int64 .npy sample "
            "with shape (3, 32, 32). Do not use YAML keys padding, pad_type, "
            "input_shape, processor_map, or weight_bits. Preserve any earlier "
            "synthesis-passing candidate as the fallback."
        )
    return result
