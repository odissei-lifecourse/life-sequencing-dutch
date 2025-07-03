import torch.nn as nn
from types import MethodType

class AttrProxy(nn.Module):
    """
    Forward attribute access to the wrapped object *and*, if that is a
    DistributedDataParallel instance, transparently look through its `.module`.

    This lets you write code that works unchanged for plain, DDP-wrapped,
    or even nested-DDP models:
        model.global_step += 1          # ok
        opt, sch = model.configure_optimizers()  # ok
    """
    def __init__(self, wrapped: nn.Module):
        super().__init__()
        object.__setattr__(self, "_wrapped", wrapped)

    # ---- helpers ---------------------------------------------------------
    def _unwrap(self):
        """Return the deepest real model (unwraps .module exactly once)."""
        inner = self._wrapped
        return getattr(inner, "module", inner)

    # ---- nn.Module API ---------------------------------------------------
    def forward(self, *args, **kwargs):
        return self._unwrap()(*args, **kwargs)

    # ---- attribute plumbing ---------------------------------------------
    def __getattr__(self, name):
        try:
            return getattr(self._wrapped, name)
        except AttributeError:
            # try one level deeper (DDP stores the real model in .module)
            inner = self._unwrap()
            if inner is not self._wrapped and hasattr(inner, name):
                return getattr(inner, name)
            raise

    def __setattr__(self, name, value):
        if name == "_wrapped":
            object.__setattr__(self, name, value)
            return

        if hasattr(self._wrapped, name):
            setattr(self._wrapped, name, value)
        elif hasattr(self._unwrap(), name):
            setattr(self._unwrap(), name, value)
        else:
            raise AttributeError(name)

    def __delattr__(self, name):
        if hasattr(self._wrapped, name):
            delattr(self._wrapped, name)
        elif hasattr(self._unwrap(), name):
            delattr(self._unwrap(), name)
        else:
            raise AttributeError(name)

    # make dir() output nicer
    def __dir__(self):
        attrs = set(super().__dir__())
        attrs.update(dir(self._unwrap()))
        return list(attrs)
