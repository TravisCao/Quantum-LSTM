"""qlstm: quantum long short-term memory layers for PyTorch.

This package provides :class:`~qlstm.QLSTM` and :class:`~qlstm.LQLSTM`, quantum
and linear-enhanced quantum LSTM layers with an API that mirrors
:class:`torch.nn.LSTM`. They implement the model from:

    Cao, Y., Zhou, X., Fei, X., Zhao, H., Liu, W., & Zhao, J. (2023).
    Linear-layer-enhanced quantum long short-term memory for carbon price
    forecasting. Quantum Machine Intelligence, 5(2), 26.
    https://doi.org/10.1007/s42484-023-00115-2
"""

from __future__ import annotations

from .cell import QLSTMCell
from .layer import LQLSTM, QLSTM
from .vqc import make_vqc

try:  # populated from installed package metadata
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("qlstm")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.1.0"

__all__ = ["QLSTM", "LQLSTM", "QLSTMCell", "make_vqc", "__version__"]
