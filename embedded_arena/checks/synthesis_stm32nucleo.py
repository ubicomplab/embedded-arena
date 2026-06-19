import json
import os
import re
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Any, NamedTuple

from exceptions import ExperimentSetupError
from schemas import RunState, CheckResult
from pydantic import BaseModel
from checks.common import (
    compile_python_if_needed,
    executable_from_env,
    existing_file,
    run_host_command,
    weighted_result,
)


class Input(BaseModel):
    model_path: str


class BoardMemoryLimits(NamedTuple):
    label: str
    external_flash_bytes: int
    model_partition_bytes: int
    activation_ram_bytes: int


STM32_BOARD_LIMITS = {
    "NUCLEON657X0Q": BoardMemoryLimits(
        label="NUCLEO-N657X0-Q / STM32N657X0H3Q",
        external_flash_bytes=64 * 1024 * 1024,
        model_partition_bytes=16 * 1024 * 1024,
        activation_ram_bytes=400 * 1024,
    ),
    "STM32N657X0": BoardMemoryLimits(
        label="NUCLEO-N657X0-Q / STM32N657X0H3Q",
        external_flash_bytes=64 * 1024 * 1024,
        model_partition_bytes=16 * 1024 * 1024,
        activation_ram_bytes=400 * 1024,
    ),
    "STM32N657X0H3Q": BoardMemoryLimits(
        label="NUCLEO-N657X0-Q / STM32N657X0H3Q",
        external_flash_bytes=64 * 1024 * 1024,
        model_partition_bytes=16 * 1024 * 1024,
        activation_ram_bytes=400 * 1024,
    ),
}


def normalized_target_name(target: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", target.upper())


def configured_memory_limits(target: str) -> BoardMemoryLimits:
    external_flash = os.environ.get("STM32_EXTERNAL_FLASH_BYTES")
    model_partition = os.environ.get("STM32_MODEL_PARTITION_BYTES")
    activation_ram = os.environ.get("STM32_ACTIVATION_RAM_BYTES")
    legacy_flash = os.environ.get("STM32_FLASH_BYTES")
    legacy_ram = os.environ.get("STM32_RAM_BYTES")
    if legacy_flash or legacy_ram:
        raise ExperimentSetupError(
            "STM32_FLASH_BYTES/STM32_RAM_BYTES are no longer supported. Use "
            "STM32_EXTERNAL_FLASH_BYTES, STM32_MODEL_PARTITION_BYTES, and "
            "STM32_ACTIVATION_RAM_BYTES."
        )
    if external_flash or model_partition or activation_ram:
        if not external_flash or not model_partition or not activation_ram:
            raise ExperimentSetupError(
                "Set STM32_EXTERNAL_FLASH_BYTES, STM32_MODEL_PARTITION_BYTES, "
                "and STM32_ACTIVATION_RAM_BYTES together, or set none of them."
            )
        try:
            return BoardMemoryLimits(
                label=f"custom STM32 target {target}",
                external_flash_bytes=int(external_flash),
                model_partition_bytes=int(model_partition),
                activation_ram_bytes=int(activation_ram),
            )
        except ValueError as exc:
            raise ExperimentSetupError(
                "STM32 memory limit environment variables must be integer byte counts."
            ) from exc

    normalized = normalized_target_name(target)
    limits = STM32_BOARD_LIMITS.get(normalized)
    if limits is None:
        supported = ", ".join(sorted(STM32_BOARD_LIMITS))
        raise ExperimentSetupError(
            f"No STM32 memory limits configured for STM32_TARGET={target!r}. "
            f"Supported built-in targets: {supported}. Set STM32_EXTERNAL_FLASH_BYTES, "
            "STM32_MODEL_PARTITION_BYTES, and STM32_ACTIVATION_RAM_BYTES for a "
            "different Nucleo board."
        )
    return limits


def memory_headroom_fraction() -> float:
    raw = os.environ.get("STM32_MEMORY_HEADROOM_FRACTION", "0.90")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ExperimentSetupError(
            "STM32_MEMORY_HEADROOM_FRACTION must be a number in (0, 1]."
        ) from exc
    if value <= 0.0 or value > 1.0:
        raise ExperimentSetupError(
            "STM32_MEMORY_HEADROOM_FRACTION must be a number in (0, 1]."
        )
    return value


def parse_int_with_commas(value: str) -> int:
    return int(value.replace(",", ""))


def parse_byte_quantity(value: str, unit: str) -> int:
    scale = {
        "B": 1,
        "BYTE": 1,
        "BYTES": 1,
        "KB": 1000,
        "KIB": 1024,
        "MB": 1000 * 1000,
        "MIB": 1024 * 1024,
    }
    normalized_unit = unit.upper()
    if normalized_unit not in scale:
        raise ValueError(f"unsupported byte unit: {unit}")
    return int(float(value.replace(",", "")) * scale[normalized_unit])


def parse_named_byte_line(output: str, name: str) -> int | None:
    pattern = rf"^\s*{re.escape(name)}\s*:\s*([0-9,.]+)\s*(B|Bytes|KiB|KB|MiB|MB)\b"
    match = re.search(pattern, output, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return None
    return parse_byte_quantity(match.group(1), match.group(2))


def parse_stm32_memory_usage(output: str) -> dict[str, int] | None:
    """Return generated FLASH/RAM use in bytes from STM32Cube.AI output."""
    summary_match = list(re.finditer(r"^\s*Summary\s+-", output, flags=re.MULTILINE))
    if summary_match:
        summary = output[summary_match[-1].start() :]
        for line in summary.splitlines():
            match = re.match(r"^\s*TOTAL\s+([0-9,]+)\s+([0-9,]+)\s*$", line)
            if match:
                return {
                    "flash_bytes": parse_int_with_commas(match.group(1)),
                    "ram_bytes": parse_int_with_commas(match.group(2)),
                }

    requested_totals = list(
        re.finditer(
            r"^\s*TOTAL\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s+([0-9,]+)\s*$",
            output,
            flags=re.MULTILINE,
        )
    )
    if requested_totals:
        match = requested_totals[-1]
        text_bytes = parse_int_with_commas(match.group(1))
        rodata_bytes = parse_int_with_commas(match.group(2))
        data_bytes = parse_int_with_commas(match.group(3))
        bss_bytes = parse_int_with_commas(match.group(4))
        return {
            "flash_bytes": text_bytes + rodata_bytes,
            "ram_bytes": data_bytes + bss_bytes,
        }

    weights = parse_named_byte_line(output, "weights (ro)")
    ram_total = parse_named_byte_line(output, "ram (total)")
    activations = parse_named_byte_line(output, "activations (rw)")
    if weights is None and ram_total is None and activations is None:
        return None
    return {
        "flash_bytes": weights or 0,
        "ram_bytes": ram_total if ram_total is not None else (activations or 0),
    }


def load_c_info(output_dir: Path) -> tuple[dict[str, Any] | None, str]:
    candidates = [
        output_dir / "network_c_info.json",
        output_dir / "workspace" / "neural_art__network" / "c_info.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text()), str(path)
        except json.JSONDecodeError as exc:
            return None, f"invalid JSON in {path}: {exc}"
    return None, "missing network_c_info.json/c_info.json"


def count_node_mappings(c_info: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(node: dict[str, Any]) -> None:
        mapping = node.get("mapping")
        if isinstance(mapping, str):
            counts[mapping] = counts.get(mapping, 0) + 1
        for child in node.get("subgraph_nodes") or []:
            if isinstance(child, dict):
                visit(child)

    for graph in c_info.get("graphs") or []:
        if not isinstance(graph, dict):
            continue
        for node in graph.get("nodes") or []:
            if isinstance(node, dict):
                visit(node)
    return counts


def memory_pool_status(c_info: dict[str, Any]) -> tuple[bool, str, dict[str, int]]:
    expected_names = {
        "cpuRAM2",
        "npuRAM3",
        "npuRAM4",
        "npuRAM5",
        "npuRAM6",
        "cpuRAM2_npuRAM3_npuRAM4_npuRAM5_npuRAM6",
        "octoFlash",
    }
    pools = c_info.get("memory_pools")
    if not isinstance(pools, list) or not pools:
        return False, "c_info does not contain memory_pools", {}
    unexpected: list[str] = []
    overfull: list[str] = []
    usage: dict[str, int] = {}
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        name = str(pool.get("name") or "")
        used = int(pool.get("used_size_bytes") or 0)
        size = int(pool.get("size_bytes") or 0)
        usage[name] = used
        if name and name not in expected_names:
            unexpected.append(name)
        if size >= 0 and used > size:
            overfull.append(f"{name}: used {used} > size {size}")
    if unexpected or overfull:
        detail = []
        if unexpected:
            detail.append(f"unexpected pools: {sorted(set(unexpected))}")
        if overfull:
            detail.append("overfull pools: " + "; ".join(overfull))
        return False, "; ".join(detail), usage
    return (
        True,
        "N657 memory pools used: "
        + ", ".join(
            f"{name}={format_bytes(usage.get(name, 0))}"
            for name in sorted(expected_names)
        ),
        usage,
    )


def generated_blob_status(
    output_dir: Path, c_info: dict[str, Any], limits: BoardMemoryLimits
) -> tuple[bool, str]:
    pools = c_info.get("memory_pools")
    if not isinstance(pools, list):
        return False, "c_info does not contain memory_pools"
    issues: list[str] = []
    details: list[str] = []
    for pool in pools:
        if not isinstance(pool, dict):
            continue
        fname = pool.get("fname")
        if not isinstance(fname, str) or not fname:
            continue
        used = int(pool.get("used_size_bytes") or 0)
        size = int(pool.get("size_bytes") or 0)
        blob_paths = [
            output_dir / f"network_{fname}",
            output_dir / "workspace" / "neural_art__network" / fname,
        ]
        existing = next((path for path in blob_paths if path.exists()), None)
        if used > 0 and existing is None:
            issues.append(f"{fname}: used {used} bytes but generated blob is missing")
            continue
        if existing is None:
            continue
        actual = existing.stat().st_size
        details.append(f"{fname}={format_bytes(actual)}")
        if actual > size:
            issues.append(f"{fname}: blob {actual} bytes exceeds pool {size} bytes")
        if "xSPI" in fname or "octo" in fname:
            if actual > limits.model_partition_bytes:
                issues.append(
                    f"{fname}: blob {format_bytes(actual)} exceeds model partition "
                    f"{format_bytes(limits.model_partition_bytes)}"
                )
    if issues:
        return False, "; ".join(issues)
    return True, "generated Neural-ART raw blobs fit pools: " + ", ".join(details)


def io_tensor_status(c_info: dict[str, Any]) -> tuple[bool, str]:
    supported_types = {"FLOAT", "INT8", "UINT8", "BOOL"}
    originals: list[dict[str, Any]] = []
    for graph in c_info.get("graphs") or []:
        if not isinstance(graph, dict):
            continue
        for key in ("original_inputs", "original_outputs"):
            values = graph.get(key) or []
            if isinstance(values, list):
                originals.extend(item for item in values if isinstance(item, dict))
    if not originals:
        return False, "c_info does not report original input/output tensors"
    issues: list[str] = []
    details: list[str] = []
    for tensor in originals:
        name = tensor.get("name") or "<unnamed>"
        shape = tensor.get("shape") or []
        data_format = tensor.get("data_format") or {}
        dtype = data_format.get("type")
        size = data_format.get("size")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim <= 0 for dim in shape
        ):
            issues.append(f"{name}: non-static shape {shape}")
        if dtype not in supported_types:
            issues.append(f"{name}: unsupported IO dtype {dtype}")
        if dtype in {"INT8", "UINT8"} and size != 8:
            issues.append(f"{name}: integer IO is {size} bits, expected 8")
        details.append(f"{name}:{dtype}{size} shape={shape}")
    if issues:
        return False, "; ".join(issues)
    return True, "static supported IO tensors: " + "; ".join(details)


def inspect_deployable_model(path: Path) -> tuple[bool, str]:
    """Check deployable artifact shape properties that matter for embedded builds."""
    if path.suffix.lower() == ".onnx":
        script = textwrap.dedent("""
            import json
            import sys

            import onnx

            model = onnx.load(sys.argv[1])
            onnx.checker.check_model(model)
            dynamic = []
            inputs = []
            for value_info in model.graph.input:
                dims = []
                for dim in value_info.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        dims.append(str(dim.dim_value))
                    else:
                        dynamic.append(value_info.name)
                        dims.append(dim.dim_param or "?")
                inputs.append(f"{value_info.name}=[{', '.join(dims)}]")
            print(json.dumps({"dynamic": sorted(set(dynamic)), "inputs": inputs}))
            """)
        ok, output = run_host_command(
            [onnx_export_python(), "-c", script, str(path)],
            timeout_seconds=int(
                os.environ.get("STM32_ONNX_INSPECT_TIMEOUT_SECONDS", "60")
            ),
        )
        if not ok:
            return (
                False,
                f"ONNX model failed validation/static-shape inspection: {output}",
            )
        try:
            payload = json.loads(output.splitlines()[-1])
        except Exception as exc:
            return False, f"could not parse ONNX inspection output: {exc}: {output}"
        dynamic = payload.get("dynamic") or []
        if dynamic:
            return (
                False,
                "ONNX inputs must have fully static dimensions for STM32N657 "
                f"synthesis; dynamic inputs: {dynamic}",
            )
        return True, "ONNX model is valid with static inputs: " + "; ".join(
            payload.get("inputs") or []
        )

    if path.suffix.lower() == ".tflite":
        return (
            True,
            "TFLite artifact accepted; shape inspection is delegated to STEdgeAI",
        )

    return False, f"unsupported deployable model extension: {path.suffix}"


def write_nucleo_n657x0q_memory_pool(path: Path, limits: BoardMemoryLimits) -> None:
    """Write the NUCLEO-N657X0-Q app-style memory map used by ST's N6 examples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    activation_kib = max(1, limits.activation_ram_bytes // 1024)
    model_mib = max(1, limits.model_partition_bytes // (1024 * 1024))
    payload: dict[str, Any] = {
        "params": {
            "param": [
                {
                    "paramname": "max_onchip_sram_size",
                    "value": str(activation_kib),
                    "magnitude": "KBYTES",
                }
            ]
        },
        "memory": {
            "cacheinfo": [
                {
                    "nlines": 512,
                    "linesize": 64,
                    "associativity": 8,
                    "bypass_enable": 1,
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "MID",
                        "latency": "MID",
                        "byteWidth": 8,
                        "freqRatio": 2.50,
                    },
                }
            ],
            "mem_file_prefix": "atonbuf",
            "mempools": [
                {
                    "fname": "AXISRAM2",
                    "name": "cpuRAM2",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "MID",
                        "latency": "MID",
                        "byteWidth": 8,
                        "freqRatio": 2.50,
                    },
                    "offset": {"value": "0x3419c000", "magnitude": "BYTES"},
                    "size": {"value": str(activation_kib), "magnitude": "KBYTES"},
                },
                {
                    "fname": "AXISRAM3",
                    "name": "npuRAM3",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "HIGH",
                        "latency": "LOW",
                        "byteWidth": 8,
                        "freqRatio": 1.25,
                    },
                    "offset": {"value": "0x34200000", "magnitude": "BYTES"},
                    "size": {"value": "448", "magnitude": "KBYTES"},
                },
                {
                    "fname": "AXISRAM4",
                    "name": "npuRAM4",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "HIGH",
                        "latency": "LOW",
                        "byteWidth": 8,
                        "freqRatio": 1.25,
                    },
                    "offset": {"value": "0x34270000", "magnitude": "BYTES"},
                    "size": {"value": "448", "magnitude": "KBYTES"},
                },
                {
                    "fname": "AXISRAM5",
                    "name": "npuRAM5",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "HIGH",
                        "latency": "LOW",
                        "byteWidth": 8,
                        "freqRatio": 1.25,
                    },
                    "offset": {"value": "0x342e0000", "magnitude": "BYTES"},
                    "size": {"value": "448", "magnitude": "KBYTES"},
                },
                {
                    "fname": "AXISRAM6",
                    "name": "npuRAM6",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_WRITE",
                        "throughput": "HIGH",
                        "latency": "LOW",
                        "byteWidth": 8,
                        "freqRatio": 1.25,
                    },
                    "offset": {"value": "0x34350000", "magnitude": "BYTES"},
                    "size": {"value": "448", "magnitude": "KBYTES"},
                },
                {
                    "fname": "xSPI2",
                    "name": "octoFlash",
                    "fformat": "FORMAT_RAW",
                    "prop": {
                        "rights": "ACC_READ",
                        "throughput": "MID",
                        "latency": "HIGH",
                        "byteWidth": 1,
                        "freqRatio": 6.00,
                        "cacheable": "CACHEABLE_ON",
                        "constants_preferred": "true",
                    },
                    "offset": {"value": "0x70380000", "magnitude": "BYTES"},
                    "size": {"value": str(model_mib), "magnitude": "MBYTES"},
                },
            ],
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_neural_art_profile(path: Path, memory_pool_path: Path) -> str:
    profile_name = "nucleo-n657x0q-strict"
    payload = {
        "Globals": {},
        "Profiles": {
            profile_name: {
                "memory_pool": str(memory_pool_path),
                "options": (
                    "--optimization 3 --all-buffers-info --mvei "
                    "--cache-maintenance --Oauto-sched --native-float "
                    "--enable-virtual-mem-pools --Omax-ca-pipe 4 --Ocache-opt --Os"
                ),
            }
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return f"{profile_name}@{path}"


def format_bytes(value: int) -> str:
    return f"{value:,} B"


def ensure_arm_toolchain_path() -> None:
    arm_bin = os.environ.get("ARM_GNU_TOOLCHAIN_BIN")
    if arm_bin:
        current_path = os.environ.get("PATH", "")
        paths = current_path.split(os.pathsep) if current_path else []
        if arm_bin not in paths:
            os.environ["PATH"] = arm_bin + (
                os.pathsep + current_path if current_path else ""
            )

    if shutil.which("arm-none-eabi-gcc") is None:
        raise ExperimentSetupError(
            "arm-none-eabi-gcc is not on PATH. Run scripts/setup_max78000.sh or "
            "install Arm GNU Toolchain, then set ARM_GNU_TOOLCHAIN_BIN in .env."
        )


def onnx_export_python() -> str:
    return (
        os.environ.get("STM32_ONNX_PYTHON")
        or os.environ.get("AI8X_PYTHON")
        or sys.executable
    )


def export_python_model_to_onnx(
    state: RunState,
    *,
    model_path: Path,
    relative_model_path: str,
) -> tuple[Path | None, str | None]:
    output_dir = Path(state.sandbox.sandbox_path) / ".generated" / "stm32"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}.onnx"
    script_path = output_dir / f"export_{model_path.stem}.py"
    script = textwrap.dedent(f"""
        from __future__ import annotations

        import importlib.util
        import inspect
        import pathlib
        import sys

        import torch

        model_path = pathlib.Path({str(model_path)!r}).resolve()
        output_path = pathlib.Path({str(output_path)!r}).resolve()
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
        model = factory().eval()

        candidates = [
            (1, 16000),
            (1, 1, 16000),
            (1, 1024),
            (1, 10),
            (1, 3, 32, 32),
            (1, 1, 32, 32),
        ]
        errors = []
        sample = None
        output = None
        with torch.no_grad():
            for shape in candidates:
                x = torch.zeros(shape, dtype=torch.float32)
                try:
                    y = model(x)
                except Exception as exc:
                    errors.append(f"{{shape}}: {{exc}}")
                    continue
                if not torch.is_tensor(y):
                    errors.append(f"{{shape}}: output is not a tensor")
                    continue
                sample = x
                output = y
                break
        if sample is None:
            raise SystemExit("Could not find an export input shape for {relative_model_path}: " + "; ".join(errors[:6]))

        torch.onnx.export(
            model,
            sample,
            output_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=13,
            do_constant_folding=True,
        )
        print(f"exported {{output_path}} input_shape={{tuple(sample.shape)}} output_shape={{tuple(output.shape)}}")
        """)
    script_path.write_text(script)
    ok, output = run_host_command(
        [onnx_export_python(), str(script_path)],
        cwd=Path(state.sandbox.sandbox_path),
        timeout_seconds=int(os.environ.get("STM32_ONNX_EXPORT_TIMEOUT_SECONDS", "180")),
    )
    if not ok:
        return None, output
    if not output_path.exists() or output_path.stat().st_size == 0:
        return None, f"ONNX export did not create {output_path}"
    return output_path, output


def check(state: RunState, input: Input) -> CheckResult:
    """Checks that the model can be synthesized to run on the STM32N"""
    ensure_arm_toolchain_path()

    model_path, error = existing_file(state, input.model_path)
    if error:
        return CheckResult(
            success=False, score=0.0, score_unit="fraction", feedback=error
        )
    assert model_path is not None

    checks = [
        ("model file exists", True, 0.25, str(model_path)),
    ]
    compile_ok, compile_detail = compile_python_if_needed(state, input.model_path)
    checks.append(
        ("model file compiles when applicable", compile_ok, 0.10, compile_detail)
    )
    if not all(passed for _, passed, _, _ in checks):
        return weighted_result(checks=checks, success_threshold=1.0)

    deployable_model_path = model_path
    if model_path.suffix.lower() == ".py":
        generated_path, export_detail = export_python_model_to_onnx(
            state,
            model_path=model_path,
            relative_model_path=input.model_path,
        )
        checks.append(
            (
                "generated ONNX artifact from Python model",
                generated_path is not None,
                0.25,
                export_detail or str(generated_path),
            )
        )
        if generated_path is None:
            return weighted_result(checks=checks, success_threshold=1.0)
        deployable_model_path = generated_path

    extension_ok = deployable_model_path.suffix.lower() in {".tflite", ".onnx"}
    checks.append(
        (
            "STM32Cube.AI-supported model artifact extension",
            extension_ok,
            0.25,
            deployable_model_path.suffix
            or "no extension; export to ONNX or TFLite for STM32Cube.AI",
        )
    )
    if not extension_ok:
        return weighted_result(checks=checks, success_threshold=1.0)

    target = os.environ.get("STM32_TARGET", "NUCLEO-N657X0-Q")
    limits = configured_memory_limits(target)
    headroom = memory_headroom_fraction()
    model_partition_budget = int(limits.model_partition_bytes * headroom)
    activation_ram_budget = int(limits.activation_ram_bytes * headroom)
    recommendations: list[str] = []
    artifact_size = deployable_model_path.stat().st_size
    checks.append(
        (
            f"{limits.label} deployable artifact fits model partition",
            artifact_size <= model_partition_budget,
            0.10,
            (
                f"artifact={format_bytes(artifact_size)}; budget "
                f"{format_bytes(model_partition_budget)} of "
                f"{format_bytes(limits.model_partition_bytes)} model partition "
                f"at {headroom:.0%} headroom"
            ),
        )
    )
    inspect_ok, inspect_detail = inspect_deployable_model(deployable_model_path)
    checks.append(
        (
            "deployable model has embedded-compatible static shape",
            inspect_ok,
            0.15,
            inspect_detail,
        )
    )
    if not inspect_ok:
        return weighted_result(checks=checks, success_threshold=1.0)

    checker = executable_from_env("STM32AI_COMMAND", ["stedgeai", "stm32ai"])
    if checker is None:
        raise ExperimentSetupError(
            "STM32Cube.AI/STEdgeAI-Core command is not configured. Run scripts/setup_stm32ai.sh, then source .env or export STM32AI_COMMAND."
        )
    checker_path = Path(checker)
    if checker_path.exists() and not os.access(checker_path, os.X_OK):
        raise ExperimentSetupError(
            f"STM32AI_COMMAND is not executable: {checker}. Run scripts/setup_stm32ai.sh again."
        )

    output_dir = (
        Path(state.sandbox.sandbox_path) / ".stm32ai" / deployable_model_path.stem
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = output_dir / "workspace"
    memory_pool_path = output_dir / "nucleo_n657x0q.mpool"
    write_nucleo_n657x0q_memory_pool(memory_pool_path, limits)
    neural_art_profile_path = output_dir / "nucleo_n657x0q_neural_art.json"
    neural_art_profile = write_neural_art_profile(
        neural_art_profile_path, memory_pool_path
    )
    name = Path(checker).name.lower()
    if "stedgeai" in name:
        command = [
            checker,
            "generate",
            "--model",
            str(deployable_model_path),
            "--target",
            "stm32n6",
            "--output",
            str(output_dir),
            "--workspace",
            str(workspace_dir),
            "--st-neural-art",
            neural_art_profile,
            "--quiet",
        ]
    else:
        command = [
            checker,
            "generate",
            "-m",
            str(deployable_model_path),
            "-o",
            str(output_dir),
            "--target",
            "stm32n6",
            "--st-neural-art",
            neural_art_profile,
        ]
    synth_ok, synth_output = run_host_command(
        command,
        timeout_seconds=int(os.environ.get("STM32AI_TIMEOUT_SECONDS", "600")),
    )
    checks.append(
        (
            "STM32Cube.AI code generation",
            synth_ok,
            0.40,
            (
                synth_output[-6000:]
                if synth_output
                else "STM32Cube.AI generation completed"
            ),
        )
    )
    if synth_ok:
        generated = output_dir / "network.c"
        usage = parse_stm32_memory_usage(synth_output)
        neural_art_used = (
            "EXECUTING NEURAL ART COMPILER" in synth_output
            and "target/series" in synth_output
            and "stm32n6npu" in synth_output
        )
        checks.append(
            (
                "STM32N6 Neural-ART compiler path used",
                neural_art_used,
                0.25,
                (
                    "STEdgeAI invoked atonn and reported target/series stm32n6npu"
                    if neural_art_used
                    else "missing Neural-ART/atonn stm32n6npu evidence in generation output"
                ),
            )
        )
        c_info, c_info_detail = load_c_info(output_dir)
        checks.append(
            (
                "Neural-ART c_info metadata generated",
                c_info is not None,
                0.15,
                c_info_detail,
            )
        )
        if c_info is not None:
            environment = c_info.get("environment") or {}
            target_info = (
                str(environment.get("target") or "")
                + " "
                + str((c_info.get("memory_footprint") or {}).get("series") or "")
            ).lower()
            target_ok = "stm32n6" in target_info or "stm32n6npu" in target_info
            checks.append(
                (
                    "c_info target is STM32N6/NPU",
                    target_ok,
                    0.15,
                    target_info or "missing c_info target/series",
                )
            )
            mapping_counts = count_node_mappings(c_info)
            hw_count = mapping_counts.get("NODE_HW", 0)
            checks.append(
                (
                    "model contains Neural-ART hardware-mapped epochs",
                    hw_count > 0,
                    0.20,
                    "node mappings: "
                    + ", ".join(
                        f"{name}={count}"
                        for name, count in sorted(mapping_counts.items())
                    ),
                )
            )
            pools_ok, pools_detail, pool_usage = memory_pool_status(c_info)
            checks.append(
                (
                    "generated model uses NUCLEO-N657X0-Q memory pools",
                    pools_ok,
                    0.20,
                    pools_detail,
                )
            )
            blob_ok, blob_detail = generated_blob_status(output_dir, c_info, limits)
            checks.append(
                (
                    "generated Neural-ART blobs fit declared pools",
                    blob_ok,
                    0.20,
                    blob_detail,
                )
            )
            io_ok, io_detail = io_tensor_status(c_info)
            checks.append(
                (
                    "generated model IO is static and supported on target",
                    io_ok,
                    0.15,
                    io_detail,
                )
            )
            if pool_usage:
                npu_activation = sum(
                    used
                    for name, used in pool_usage.items()
                    if name in {"npuRAM3", "npuRAM4", "npuRAM5", "npuRAM6"}
                )
                cpu_activation = pool_usage.get("cpuRAM2", 0)
                octoflash = pool_usage.get("octoFlash", 0)
                checks.append(
                    (
                        f"{limits.label} NPU SRAM pool budget",
                        npu_activation <= 4 * 448 * 1024,
                        0.15,
                        (
                            f"NPU SRAM pools use {format_bytes(npu_activation)}; "
                            "available 1,835,008 B across AXISRAM3-6; "
                            f"CPU app pool uses {format_bytes(cpu_activation)}; "
                            f"xSPI model blob uses {format_bytes(octoflash)}"
                        ),
                    )
                )
        checks.append(
            (
                "generated STM32 C source exists",
                generated.exists(),
                0.10,
                str(generated) if generated.exists() else f"missing {generated}",
            )
        )
        checks.append(
            (
                "STM32Cube.AI memory report parsed",
                usage is not None,
                0.10,
                (
                    f"flash={format_bytes(usage['flash_bytes'])} "
                    f"ram={format_bytes(usage['ram_bytes'])}"
                    if usage is not None
                    else "could not find FLASH/RAM TOTAL line in STM32Cube.AI output"
                ),
            )
        )
        if usage is not None:
            activation_ok = usage["ram_bytes"] <= activation_ram_budget
            if not activation_ok:
                over_by = usage["ram_bytes"] - activation_ram_budget
                recommendations.append(
                    "recommendation=activation_sram_over_budget; this is the "
                    "blocking deployability issue. Reduce activation/IO memory before "
                    "optimizing accuracy: shorten the temporal sequence early with "
                    "larger stride or pooling, reduce output frame count, reduce "
                    "intermediate channels, avoid keeping full-resolution feature maps, "
                    "and for 16000-sample speech inputs aim for roughly 20-30 output "
                    "frames or fewer before the final vocabulary projection. Return a "
                    "plain tensor from forward(), not a custom loss tuple or "
                    "HuggingFace output object. Target activations below "
                    f"{format_bytes(activation_ram_budget)} "
                    f"(currently {format_bytes(usage['ram_bytes'])}, "
                    f"{format_bytes(over_by)} over budget). If a previous candidate "
                    "passed synthesis, keep it as the fallback and only replace it "
                    "with a candidate that also passes this activation budget."
                )
            checks.append(
                (
                    f"{limits.label} model partition budget",
                    usage["flash_bytes"] <= model_partition_budget,
                    0.20,
                    (
                        f"weights/code use {format_bytes(usage['flash_bytes'])}; "
                        f"budget {format_bytes(model_partition_budget)} of "
                        f"{format_bytes(limits.model_partition_bytes)} model partition "
                        f"at {headroom:.0%} headroom; external flash total "
                        f"{format_bytes(limits.external_flash_bytes)}"
                    ),
                )
            )
            checks.append(
                (
                    f"{limits.label} activation SRAM budget",
                    activation_ok,
                    0.20,
                    (
                        f"activations/IO use {format_bytes(usage['ram_bytes'])}; "
                        f"budget {format_bytes(activation_ram_budget)} of "
                        f"{format_bytes(limits.activation_ram_bytes)} app activation SRAM "
                        f"at {headroom:.0%} headroom"
                    ),
                )
            )
    result = weighted_result(checks=checks, success_threshold=1.0)
    if result.success:
        recommendations.append(
            "recommendation=synthesis_passed; preserve this exact model as the "
            "deployable fallback. Future changes should keep activation/IO SRAM "
            "under budget; if a new candidate fails synthesis, resubmit this passing "
            "artifact while improving training settings."
        )
    else:
        recommendations.append(
            "recommendation=known_good_stm32n6_baseline; if you need to re-establish "
            "a deployable starting point, use this exact plain PyTorch architecture: "
            "input tensor shape Bx16000 or Bx1x16000, four Conv1d+ReLU blocks "
            "(1->8 kernel=15 stride=8 padding=7), "
            "(8->16 kernel=11 stride=5 padding=5), "
            "(16->24 kernel=7 stride=4 padding=3), "
            "(24->32 kernel=5 stride=4 padding=2), then Conv1d(32->79, "
            "kernel=1), and return logits.transpose(1, 2) as a plain BxTx79 "
            "tensor. Do not return loss tuples or HuggingFace output objects. "
            "Do not add BatchNorm, ReduceMean/global pooling, GRU/LSTM, attention, "
            "or Transformer layers to the fallback. This exact baseline was verified "
            "with STEdgeAI for NUCLEO-N657X0-Q / STM32N657X0H3Q: ONNX output "
            "shape [1, 25, 79], weights/code about 83 kB, activations/IO about "
            "128 kB, below the 368,640 B strict app activation budget."
        )
    if recommendations:
        result.feedback = result.feedback + "\n" + "\n".join(recommendations)  # type: ignore
    return result
