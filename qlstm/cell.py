"""Quantum LSTM cells.

A :class:`QLSTMCell` replaces the four affine gates of a classical LSTM cell
with variational quantum circuits (VQCs). Two variants are provided through the
``linear_enhanced`` flag:

* ``linear_enhanced=True`` (default) -- the **L-QLSTM** of Cao et al. (2023):
  a classical linear layer projects ``[x_t, h_{t-1}]`` down to ``n_qubits``, the
  VQC processes it, and a second linear layer projects the measurements up to
  ``hidden_size``. Input and hidden sizes are therefore unconstrained.
* ``linear_enhanced=False`` -- the classic QLSTM of Chen et al. (2020): the VQC
  acts directly on ``[x_t, h_{t-1}]``, which requires
  ``n_qubits == input_size + hidden_size`` and ``hidden_size <= n_qubits``.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from .vqc import Ansatz, Rotation, make_vqc

# "candidate" is the cell-candidate gate (g_t). It is deliberately not named
# "update": nn.ModuleDict reserves ``update`` as a method, so a child module of
# that name would collide.
_GATES = ("forget", "input", "candidate", "output")

_ACTIVATIONS = {
    "arctan": torch.arctan,
    "tanh": torch.tanh,
    "none": None,
}

InputActivation = Union[str, Callable[[torch.Tensor], torch.Tensor], None]


def _resolve_activation(activation: InputActivation) -> Optional[Callable]:
    if activation is None or callable(activation):
        return activation
    try:
        return _ACTIVATIONS[activation]
    except KeyError:
        raise ValueError(
            f"unknown input_activation {activation!r}; "
            f"use one of {sorted(_ACTIVATIONS)} or a callable"
        ) from None


class QLSTMCell(nn.Module):
    """A single quantum LSTM step.

    Args:
        input_size: Size of each input vector ``x_t``.
        hidden_size: Size of the hidden and cell states.
        n_qubits: Number of wires in each gate circuit.
        n_qlayers: Depth of the entangling ansatz.
        ansatz: Entangling ansatz, ``"basic"`` or ``"strong"``.
        rotation: Angle-embedding rotation axis.
        input_activation: Applied to the angles before embedding. Defaults to
            ``"arctan"``, which bounds the encoded angles as in Cao et al.
            (2023). Use ``"tanh"``, ``None``/``"none"``, or any callable.
        backend: PennyLane device name.
        diff_method: PennyLane differentiation method.
        linear_enhanced: See the module docstring.

    Shapes:
        * input: ``x_t`` of shape ``(batch, input_size)`` and a state tuple
          ``(h, c)`` each of shape ``(batch, hidden_size)``.
        * output: the new ``(h, c)`` each of shape ``(batch, hidden_size)``.
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
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.linear_enhanced = linear_enhanced
        self._activation = _resolve_activation(input_activation)

        concat_size = input_size + hidden_size
        if not linear_enhanced:
            if concat_size != n_qubits:
                raise ValueError(
                    "with linear_enhanced=False the circuit acts directly on "
                    f"[x_t, h_{{t-1}}], so n_qubits must equal "
                    f"input_size + hidden_size ({concat_size}); got {n_qubits}. "
                    "Set linear_enhanced=True to lift this constraint."
                )
            if hidden_size > n_qubits:
                raise ValueError(
                    f"hidden_size ({hidden_size}) cannot exceed n_qubits "
                    f"({n_qubits}) when linear_enhanced=False"
                )

        def new_vqc() -> nn.Module:
            return make_vqc(
                n_qubits=n_qubits,
                n_qlayers=n_qlayers,
                ansatz=ansatz,
                rotation=rotation,
                backend=backend,
                diff_method=diff_method,
            )

        self.vqc = nn.ModuleDict({g: new_vqc() for g in _GATES})
        if linear_enhanced:
            self.clayer_in = nn.ModuleDict(
                {g: nn.Linear(concat_size, n_qubits) for g in _GATES}
            )
            self.clayer_out = nn.ModuleDict(
                {g: nn.Linear(n_qubits, hidden_size) for g in _GATES}
            )

    def _gate(self, name: str, v_t: torch.Tensor) -> torch.Tensor:
        """Pre-activation value of one gate, shape ``(batch, hidden_size)``."""
        if self.linear_enhanced:
            a = self.clayer_in[name](v_t)
            if self._activation is not None:
                a = self._activation(a)
            q = self.vqc[name](a)
            return self.clayer_out[name](q)
        a = self._activation(v_t) if self._activation is not None else v_t
        q = self.vqc[name](a)
        return q[:, : self.hidden_size]

    def forward(
        self,
        x_t: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_t.dim() != 2:
            raise ValueError(
                f"expected x_t of shape (batch, input_size), got {tuple(x_t.shape)}"
            )
        batch = x_t.size(0)
        if state is None:
            h_t = x_t.new_zeros(batch, self.hidden_size)
            c_t = x_t.new_zeros(batch, self.hidden_size)
        else:
            h_t, c_t = state

        v_t = torch.cat((x_t, h_t), dim=1)
        f_t = torch.sigmoid(self._gate("forget", v_t))
        i_t = torch.sigmoid(self._gate("input", v_t))
        g_t = torch.tanh(self._gate("candidate", v_t))
        o_t = torch.sigmoid(self._gate("output", v_t))

        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_qubits={self.n_qubits}, linear_enhanced={self.linear_enhanced}"
        )
