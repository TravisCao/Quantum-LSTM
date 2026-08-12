"""Variational quantum circuit used as the gate nonlinearity in :mod:`qlstm`.

The circuit is a standard encode-then-entangle variational block built entirely
from PennyLane templates, which keeps it batching-safe and differentiable end to
end:

1. angle-embed the (classically projected) input on ``n_qubits`` wires,
2. apply an entangling ansatz with trainable weights,
3. measure :math:`\\langle Z \\rangle` on every wire.

Returning one expectation value per wire makes the block a drop-in nonlinearity
whose output lives in ``[-1, 1]``.
"""

from __future__ import annotations

from typing import Literal

import pennylane as qml

Ansatz = Literal["basic", "strong"]
Rotation = Literal["X", "Y", "Z"]


def make_vqc(
    n_qubits: int,
    n_qlayers: int = 1,
    ansatz: Ansatz = "basic",
    rotation: Rotation = "Y",
    backend: str = "default.qubit",
    diff_method: str = "backprop",
) -> qml.qnn.TorchLayer:
    """Build a variational quantum circuit wrapped as a ``torch`` layer.

    Args:
        n_qubits: Number of wires. The layer maps an ``n_qubits``-dimensional
            input to an ``n_qubits``-dimensional output.
        n_qlayers: Depth of the entangling ansatz.
        ansatz: ``"basic"`` uses :class:`~pennylane.BasicEntanglerLayers` (one
            parameter per wire per layer); ``"strong"`` uses
            :class:`~pennylane.StronglyEntanglingLayers` (three parameters per
            wire per layer, more expressive).
        rotation: Rotation axis for the angle embedding.
        backend: Any PennyLane device name, e.g. ``"default.qubit"`` or
            ``"lightning.qubit"``.
        diff_method: PennyLane differentiation method. ``"backprop"`` is exact
            and fast on ``default.qubit``; ``lightning.qubit`` is switched to
            ``"adjoint"`` automatically because it does not support backprop.

    Returns:
        A :class:`pennylane.qnn.TorchLayer` that accepts inputs of shape
        ``(batch, n_qubits)`` and returns ``(batch, n_qubits)``.
    """
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")
    if n_qlayers < 1:
        raise ValueError(f"n_qlayers must be >= 1, got {n_qlayers}")

    # lightning.qubit cannot backprop; adjoint is its exact, fast equivalent.
    if backend.startswith("lightning") and diff_method == "backprop":
        diff_method = "adjoint"

    dev = qml.device(backend, wires=n_qubits)

    if ansatz == "basic":
        weight_shapes = {"weights": (n_qlayers, n_qubits)}

        def entangle(weights):
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    elif ansatz == "strong":
        weight_shapes = {"weights": (n_qlayers, n_qubits, 3)}

        def entangle(weights):
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    else:
        raise ValueError(f"unknown ansatz {ansatz!r}; use 'basic' or 'strong'")

    @qml.qnode(dev, interface="torch", diff_method=diff_method)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation=rotation)
        entangle(weights)
        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return qml.qnn.TorchLayer(circuit, weight_shapes)
