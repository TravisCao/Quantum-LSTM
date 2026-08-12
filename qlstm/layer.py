"""Sequence-level quantum LSTM layers with an :class:`torch.nn.LSTM`-like API."""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .cell import InputActivation, QLSTMCell
from .vqc import Ansatz, Rotation

State = Tuple[torch.Tensor, torch.Tensor]


class QLSTM(nn.Module):
    """A (optionally stacked) quantum LSTM.

    The call signature and return values mirror :class:`torch.nn.LSTM` (one
    direction), so it can be dropped into existing models:

    >>> import torch
    >>> from qlstm import QLSTM
    >>> layer = QLSTM(input_size=8, hidden_size=4, n_qubits=4, num_layers=2)
    >>> x = torch.randn(6, 2, 8)            # (seq, batch, input_size)
    >>> out, (h_n, c_n) = layer(x)
    >>> out.shape, h_n.shape
    (torch.Size([6, 2, 4]), torch.Size([2, 2, 4]))

    With ``num_layers > 1`` the layers are stacked: each layer consumes the
    hidden-state sequence of the layer below, exactly as in
    :class:`torch.nn.LSTM`. ``output`` is the top layer's hidden sequence, and
    ``h_n``/``c_n`` are stacked over layers with shape
    ``(num_layers, batch, hidden_size)``.

    Args:
        input_size: Size of each input vector.
        hidden_size: Size of the hidden and cell states.
        n_qubits: Number of wires in each gate circuit.
        n_qlayers: Depth of the entangling ansatz.
        num_layers: Number of stacked recurrent layers (default ``1``).
        ansatz: Entangling ansatz, ``"basic"`` or ``"strong"``.
        rotation: Angle-embedding rotation axis.
        input_activation: Angle activation before embedding (default
            ``"arctan"``).
        backend: PennyLane device name.
        diff_method: PennyLane differentiation method.
        linear_enhanced: Use the linear-enhanced L-QLSTM (default) or the
            classic QLSTM. See :class:`~qlstm.cell.QLSTMCell`. With
            ``linear_enhanced=False`` and ``num_layers > 1`` every layer's
            circuit acts on its own concatenation, so ``input_size`` must equal
            ``hidden_size`` for the shared ``n_qubits`` to fit every layer.
        dropout: If non-zero, apply :class:`torch.nn.Dropout` with this
            probability to the output of each layer except the last, as in
            :class:`torch.nn.LSTM`.
        batch_first: If ``True``, inputs and outputs are shaped
            ``(batch, seq, feature)`` instead of ``(seq, batch, feature)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int = 4,
        n_qlayers: int = 1,
        num_layers: int = 1,
        ansatz: Ansatz = "basic",
        rotation: Rotation = "Y",
        input_activation: InputActivation = "arctan",
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        linear_enhanced: bool = True,
        dropout: float = 0.0,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if dropout > 0.0 and num_layers == 1:
            warnings.warn(
                "dropout is applied between stacked layers, so it has no effect "
                "with num_layers=1; set num_layers>1 or dropout=0.0.",
                stacklevel=2,
            )
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList(
            QLSTMCell(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                ansatz=ansatz,
                rotation=rotation,
                input_activation=input_activation,
                backend=backend,
                diff_method=diff_method,
                linear_enhanced=linear_enhanced,
            )
            for layer in range(num_layers)
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @property
    def cell(self) -> QLSTMCell:
        """The first (bottom) recurrent cell.

        Kept for convenience and backward compatibility; with ``num_layers > 1``
        use :attr:`cells` to reach the others.
        """
        return self.cells[0]

    def forward(
        self, x: torch.Tensor, hx: Optional[State] = None
    ) -> Tuple[torch.Tensor, State]:
        if x.dim() != 3:
            raise ValueError(
                "expected a 3D input (seq, batch, feature) or "
                f"(batch, seq, feature) with batch_first=True, got {tuple(x.shape)}"
            )
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch, _ = x.shape

        h0, c0 = self._init_state(x, batch, hx)

        layer_input = x
        h_n: List[torch.Tensor] = []
        c_n: List[torch.Tensor] = []
        for layer, cell in enumerate(self.cells):
            h_t, c_t = h0[layer], c0[layer]
            outputs = []
            for t in range(seq_len):
                h_t, c_t = cell(layer_input[t], (h_t, c_t))
                outputs.append(h_t)
            layer_seq = torch.stack(outputs, dim=0)
            if self.dropout is not None and layer < self.num_layers - 1:
                layer_seq = self.dropout(layer_seq)
            layer_input = layer_seq
            h_n.append(h_t)
            c_n.append(c_t)

        output = layer_input
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (torch.stack(h_n, dim=0), torch.stack(c_n, dim=0))

    def _init_state(
        self, x: torch.Tensor, batch: int, hx: Optional[State]
    ) -> State:
        """Return per-layer ``(h0, c0)``, each ``(num_layers, batch, hidden)``."""
        if hx is None:
            zeros = x.new_zeros(self.num_layers, batch, self.hidden_size)
            return zeros, zeros.clone()
        h0, c0 = hx
        if h0.dim() == 2:  # (batch, hidden): accept only for a single layer
            h0, c0 = h0.unsqueeze(0), c0.unsqueeze(0)
        if h0.size(0) != self.num_layers:
            raise ValueError(
                f"initial state has {h0.size(0)} layer(s) but the module has "
                f"num_layers={self.num_layers}"
            )
        return h0, c0

    def extra_repr(self) -> str:
        return f"num_layers={self.num_layers}, batch_first={self.batch_first}"


class LQLSTM(QLSTM):
    """Linear-layer-enhanced quantum LSTM (Cao et al., 2023).

    A thin alias of :class:`QLSTM` with ``linear_enhanced=True`` to name the
    paper's model explicitly. This is already the :class:`QLSTM` default.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["linear_enhanced"] = True
        super().__init__(*args, **kwargs)
