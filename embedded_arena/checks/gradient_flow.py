from schemas import RunState, CheckResult
from pydantic import BaseModel
from checks.common import (
    compile_python_if_needed,
    existing_file,
    run_sandbox_python,
    weighted_result,
)


class Input(BaseModel):
    model_path: str


def check(state: RunState, input: Input) -> CheckResult:
    """Verifies that the model can be trained by checking that the gradients can go all the way through"""
    model_path, error = existing_file(state, input.model_path)
    if error:
        return CheckResult(
            success=False,
            score=0.0,
            score_unit="fraction",
            feedback=error,
        )

    compile_ok, compile_detail = compile_python_if_needed(state, input.model_path)
    gradient_ok = False
    gradient_detail = "gradient smoke test requires a Python PyTorch model definition"
    if input.model_path.endswith(".py"):
        smoke = f"""
import importlib.util
import inspect
import pathlib
import sys

path = pathlib.Path({input.model_path!r}).resolve()
spec = importlib.util.spec_from_file_location("candidate_model", path)
module = importlib.util.module_from_spec(spec)
sys.modules["candidate_model"] = module
spec.loader.exec_module(module)

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is required for gradient smoke testing: {{exc}}")

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
model = factory() if callable(factory) else factory
if not hasattr(model, "parameters"):
    raise SystemExit("Model object does not expose parameters()")
params = [p for p in model.parameters() if getattr(p, "requires_grad", False)]
if not params:
    raise SystemExit("Model has no trainable parameters")
model.train()

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

errors = []
for shape in ((2, 10), (2, 1, 28, 28), (2, 3, 32, 32), (2, 16000)):
    model.zero_grad(set_to_none=True)
    x = torch.randn(*shape)
    try:
        output = model(x)
        tensor = first_tensor(output)
        if tensor is None:
            raise RuntimeError("forward did not return a tensor-like output")
        tensor.float().sum().backward()
    except Exception as exc:
        errors.append(f"shape={{shape}}: {{exc}}")
        continue
    grads = [p.grad for p in params if p.grad is not None]
    if grads and any(torch.isfinite(g).all().item() and g.abs().sum().item() > 0 for g in grads):
        print(f"trainable_parameters={{sum(p.numel() for p in params)}} input_shape={{shape}}")
        break
else:
    raise SystemExit("No tested input shape produced finite nonzero gradients: " + "; ".join(errors))
"""
        gradient_ok, gradient_detail = run_sandbox_python(state, smoke)

    return weighted_result(
        checks=[
            ("model file exists", True, 0.4, str(model_path)),
            ("model file compiles", compile_ok, 0.3, compile_detail),
            (
                "gradient smoke test",
                gradient_ok,
                0.3,
                gradient_detail,
            ),
        ],
        success_threshold=1.0,
    )
