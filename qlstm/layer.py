"""Sequence-level quantum LSTM layers with an :class:`torch.nn.LSTM`-like API."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .cell import InputActivation, QLSTMCell
from .vqc import Ansatz, Rotation

State = Tuple[torch.Tensor, torch.Tensor]


class QLSTM(nn.Module):
    """A single-layer quantum LSTM.

    The call signature and return values mirror :class:`torch.nn.LSTM` (for one
    layer, one direction), so it can be dropped into existing models:

    >>> import torch
    >>> from qlstm import QLSTM
    >>> layer = QLSTM(input_size=8, hidden_size=4, n_qubits=4)
    >>> x = torch.randn(6, 2, 8)            # (seq, batch, input_size)
    >>> out, (h_n, c_n) = layer(x)
    >>> out.shape, h_n.shape
    (torch.Size([6, 2, 4]), torch.Size([1, 2, 4]))

    Args:
        input_size: Size of each input vector.
        hidden_size: Size of the hidden and cell states.
        n_qubits: Number of wires in each gate circuit.
        n_qlayers: Depth of the entangling ansatz.
        ansatz: Entangling ansatz, ``"basic"`` or ``"strong"``.
        rotation: Angle-embedding rotation axis.
        input_activation: Angle activation before embedding (default
            ``"arctan"``).
        backend: PennyLane device name.
        diff_method: PennyLane differentiation method.
        linear_enhanced: Use the linear-enhanced L-QLSTM (default) or the
            classic QLSTM. See :class:`~qlstm.cell.QLSTMCell`.
        batch_first: If ``True``, inputs and outputs are shaped
            ``(batch, seq, feature)`` instead of ``(seq, batch, feature)``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_qubits: int = 4,
        n_qlayers: int = 1,
        ansatz: Ansatz = "basic",
        rotation: Rotation = "Y",
        input_activation: InputActivation = "arctan",
        backend: str = "default.qubit",
        diff_method: str = "backprop",
        linear_enhanced: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.cell = QLSTMCell(
            input_size=input_size,
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

        if hx is None:
            h_t = x.new_zeros(batch, self.hidden_size)
            c_t = x.new_zeros(batch, self.hidden_size)
        else:
            h_t, c_t = hx
            # Accept nn.LSTM-style (num_layers, batch, hidden) states.
            if h_t.dim() == 3:
                h_t, c_t = h_t[0], c_t[0]

        outputs = []
        for t in range(seq_len):
            h_t, c_t = self.cell(x[t], (h_t, c_t))
            outputs.append(h_t)

        output = torch.stack(outputs, dim=0)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, (h_t.unsqueeze(0), c_t.unsqueeze(0))

    def extra_repr(self) -> str:
        return f"batch_first={self.batch_first}"


class LQLSTM(QLSTM):
    """Linear-layer-enhanced quantum LSTM (Cao et al., 2023).

    A thin alias of :class:`QLSTM` with ``linear_enhanced=True`` to name the
    paper's model explicitly. This is already the :class:`QLSTM` default.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs["linear_enhanced"] = True
        super().__init__(*args, **kwargs)
