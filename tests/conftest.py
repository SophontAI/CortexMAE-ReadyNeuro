"""Stand in for Brainmarks when the optional ``eval`` extra is not installed.

``src/brainmarks/models/fmrifm.py`` imports two names from the benchmark suite. Without
them the file that covers the padding mask and the frame reflection is skipped, so on a
stock checkout the behaviour the README calls load-bearing contributes zero assertions.
The stubs are only installed when the real package is absent.
"""

import sys
import types
from collections import namedtuple


def _stub_brainmarks() -> None:
    base = types.ModuleType("brainmarks.models.base")
    base.Embeddings = namedtuple("Embeddings", "cls pooled tokens pad_mask")
    registry = types.ModuleType("brainmarks.models.registry")
    registry.register_model = lambda function: function
    sys.modules.setdefault("brainmarks.models.base", base)
    sys.modules.setdefault("brainmarks.models.registry", registry)


try:  # pragma: no cover - exercised by whichever branch the environment takes
    import brainmarks.models.base  # noqa: F401
    import brainmarks.models.registry  # noqa: F401
except ImportError:
    _stub_brainmarks()
