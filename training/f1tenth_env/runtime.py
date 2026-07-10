"""Runtime dtype/device configuration for the training environment.

These values are backend-agnostic and owned by the env layer itself, so the
torch backend never depends on Genesis. Entry points call :func:`configure`
once at startup; the Genesis backend mirrors its engine dtype/device here too.
"""

from __future__ import annotations

import torch

tc_float: torch.dtype = torch.float32
tc_int: torch.dtype = torch.int32
device: torch.device = torch.device("cpu")
EPS: float = 1e-12


def configure(
    *,
    float_dtype: torch.dtype | None = None,
    int_dtype: torch.dtype | None = None,
    dev: torch.device | str | None = None,
    eps: float | None = None,
) -> None:
    global tc_float, tc_int, device, EPS
    if float_dtype is not None:
        tc_float = float_dtype
    if int_dtype is not None:
        tc_int = int_dtype
    if dev is not None:
        device = dev if isinstance(dev, torch.device) else torch.device(dev)
    if eps is not None:
        EPS = eps
