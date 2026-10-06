"""One place to turn loose numbers into tensors without breaking the graph."""

from __future__ import annotations

import torch
from torch import Tensor

#: Guards logs and divisions against exactly-zero inputs. Small enough not to
#: perturb any price a desk would quote, large enough to keep float64 finite.
EPS = 1e-12


def as_tensor(x, dtype: torch.dtype | None = None, device=None) -> Tensor:
    """Coerce ``x`` to a tensor, passing existing tensors through untouched.

    Passing tensors through by identity is the point: a tensor that reaches here
    may be an ``nn.Parameter`` the caller intends to differentiate against, and
    rebuilding it would silently cut it out of the autograd graph.
    """
    if isinstance(x, Tensor):
        if dtype is not None and x.dtype != dtype:
            x = x.to(dtype)
        if device is not None and x.device != torch.device(device):
            x = x.to(device)
        return x
    return torch.as_tensor(x, dtype=dtype or torch.get_default_dtype(), device=device)


def interp_linear(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` over knots ``xp``, evaluated at ``x``.

    ``xp`` must be sorted and hold at least two points. Values outside the knot
    range are extrapolated along the nearest segment; callers that want another
    rule outside the knots clamp ``x`` first and patch the ends. Differentiable
    in ``x`` and ``fp``; the knot lookup itself is not, and need not be.
    """
    idx = torch.searchsorted(xp, x.detach().contiguous()).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    f0, f1 = fp[idx - 1], fp[idx]
    return f0 + (f1 - f0) * (x - x0) / (x1 - x0)
