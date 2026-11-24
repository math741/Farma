"""Hardware detection and memory accounting (spec §252-255, §419-421).

Nothing about the machine is assumed: the GPU (if any) is detected, its
memory is reported as current / reserved / peak / free, and the profile is
chosen from what is actually present.
"""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, asdict


@dataclass
class HardwareInfo:
    device: str
    gpu_name: str
    gpu_total_gb: float
    gpu_capability: str
    cpu_count: int
    ram_total_gb: float
    python: str
    torch: str
    cuda: str
    in_colab: bool
    is_a100: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _ram_total_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def in_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ


def detect(prefer: str = "auto") -> HardwareInfo:
    import torch

    use_cuda = torch.cuda.is_available() and prefer in ("auto", "cuda", "gpu")
    gpu_name, total, cap = "", 0.0, ""
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        total = props.total_memory / 1024**3
        cap = f"{props.major}.{props.minor}"
    return HardwareInfo(
        device="cuda" if use_cuda else "cpu",
        gpu_name=gpu_name,
        gpu_total_gb=round(total, 2),
        gpu_capability=cap,
        cpu_count=os.cpu_count() or 1,
        ram_total_gb=round(_ram_total_gb(), 2),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=str(torch.version.cuda) if use_cuda else "none",
        in_colab=in_colab(),
        is_a100="A100" in gpu_name,
    )


def memory_stats(device: str) -> dict:
    """Current / reserved / peak / free memory in GiB (GPU) or RAM (CPU)."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return {
            "kind": "vram",
            "current_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
            "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "free_gb": round(free / 1024**3, 3),
            "total_gb": round(total / 1024**3, 3),
        }
    rss = 0.0
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    avail = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return {"kind": "ram", "current_gb": round(rss, 3), "reserved_gb": round(rss, 3), "peak_gb": round(rss, 3),
            "free_gb": round(avail, 3), "total_gb": round(_ram_total_gb(), 3)}


def configure_determinism(device: str, deterministic: bool) -> None:
    """Enable deterministic kernels. Must run before the first CUDA op."""
    import torch

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        if device.startswith("cuda"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    if device == "cpu":
        # A fixed intra-op thread count keeps CPU reductions reproducible.
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
