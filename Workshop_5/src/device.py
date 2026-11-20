"""Device selection helpers supporting AMD ROCm (HIP), NVIDIA CUDA, DirectML, and CPU."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch

DEVICE_CHOICES = ("auto", "rocm", "cuda", "dml", "cpu")


def _try_import_directml():
    try:
        import torch_directml  # type: ignore[import-not-found]
        return torch_directml
    except ImportError:
        return None


def is_directml(device: torch.device) -> bool:
    return device.type == "privateuseone"


def is_rocm_build() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _cuda_or_rocm_label() -> str:
    gpu_name = torch.cuda.get_device_name(0)
    if is_rocm_build():
        return f"AMD ROCm/HIP ({gpu_name}, hip={torch.version.hip})"
    return f"NVIDIA CUDA ({gpu_name}, cuda={torch.version.cuda})"


def resolve_device(preferred: str = "auto") -> Tuple[torch.device, str]:
    """Resolve the compute device for training/evaluation.

    Note: PyTorch on AMD ROCm exposes GPUs (including Radeon RX 9060 XT / gfx1200)
    through the ``torch.cuda`` HIP compatibility namespace.
    """
    choice = (preferred or "auto").lower()
    if choice not in DEVICE_CHOICES:
        raise ValueError(
            f"Dispositivo no válido: {preferred!r}. Opciones: {', '.join(DEVICE_CHOICES)}"
        )

    if choice == "cpu":
        return torch.device("cpu"), "CPU"

    if choice in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "No se detectó una GPU compatible con ROCm/CUDA en PyTorch. "
                "Para AMD Radeon RX 9060 XT (gfx1200), instala PyTorch con soporte ROCm "
                "(ver README.md) y verifica con `python -m src.check_gpu`."
            )
        if choice == "rocm" and not is_rocm_build():
            raise RuntimeError(
                "Se solicitó `--device rocm`, pero esta instalación de PyTorch no tiene "
                "soporte HIP/ROCm (`torch.version.hip` es None). "
                "Instala los wheels de PyTorch para ROCm (gfx1200) o usa `--device auto`."
            )
        return torch.device("cuda"), _cuda_or_rocm_label()

    if choice == "dml":
        dml = _try_import_directml()
        if dml is None or not dml.is_available():
            raise RuntimeError(
                "Se solicitó `--device dml`, pero `torch-directml` no está instalado o no "
                "detectó una GPU DirectX 12. Instálalo con `pip install torch-directml` "
                "o usa ROCm nativo con `--device rocm` (recomendado para RX 9060 XT)."
            )
        idx = dml.default_device()
        return dml.device(idx), f"DirectML ({dml.device_name(idx)})"

    # auto: ROCm/CUDA -> DirectML -> CPU
    if torch.cuda.is_available():
        return torch.device("cuda"), _cuda_or_rocm_label()

    dml = _try_import_directml()
    if dml is not None and dml.is_available():
        idx = dml.default_device()
        return dml.device(idx), f"DirectML ({dml.device_name(idx)})"

    return torch.device("cpu"), "CPU (sin aceleración GPU detectada)"


def print_device_info(device: torch.device, description: str) -> None:
    print(f"Device: {device} | {description}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_state_dict(module: torch.nn.Module) -> dict:
    """Return a CPU copy of a module's state_dict for portable checkpoints."""
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def dataloader_kwargs(device: torch.device) -> dict:
    """Extra DataLoader kwargs safe for ROCm, CUDA, DirectML, and CPU."""
    return {"pin_memory": device.type == "cuda"}
