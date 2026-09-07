"""One place to turn loose numbers into tensors without breaking the graph."""

from __future__ import annotations

import torch
from torch import Tensor

#: Guards logs and divisions against exactly-zero inputs. Small enough not to
#: perturb any price a desk would quote, large enough to keep float64 finite.
EPS = 1e-12

#: The dtype every market object and model parameter is built in.
#:
#: Explicit rather than ``torch.get_default_dtype()``, because that is process
#: global and whatever the *caller* last set it to. Under float32 this package
#: is quietly wrong in places that matter: ``EPS`` above is below float32's
#: resolution entirely, and ``RateCurve.instantaneous_forward`` is a divided
#: difference over a 1e-4 window, which discards roughly seven of float32's
#: seven significant digits.
#:
#: Constructing in float64 costs nothing on CPU and is the safe direction:
#: :class:`~torch_pricer.pricer.engine.MCConfig` casts the whole snapshot and
#: model to ``config.dtype`` before pricing, so a caller who wants float32 for
#: speed still gets it -- from a value that was correct to begin with.
DEFAULT_DTYPE = torch.float64


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
    return torch.as_tensor(x, dtype=dtype or DEFAULT_DTYPE, device=device)
