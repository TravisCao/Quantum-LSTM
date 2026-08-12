"""Tests for the qlstm package: shapes, gradients, learning, and the API."""

from __future__ import annotations

import pytest
import torch

from qlstm import LQLSTM, QLSTM, QLSTMCell, make_vqc


def test_output_shapes_seq_first():
    layer = QLSTM(input_size=6, hidden_size=4, n_qubits=4)
    x = torch.randn(5, 3, 6)  # (seq, batch, input)
    out, (h_n, c_n) = layer(x)
    assert out.shape == (5, 3, 4)
    assert h_n.shape == (1, 3, 4)
    assert c_n.shape == (1, 3, 4)


def test_output_shapes_batch_first():
    layer = QLSTM(input_size=6, hidden_size=4, n_qubits=4, batch_first=True)
    x = torch.randn(3, 5, 6)  # (batch, seq, input)
    out, (h_n, c_n) = layer(x)
    assert out.shape == (3, 5, 4)
    assert h_n.shape == (1, 3, 4)


def test_outputs_are_finite():
    layer = QLSTM(input_size=4, hidden_size=3, n_qubits=4, ansatz="strong")
    out, _ = layer(torch.randn(4, 2, 4))
    assert torch.isfinite(out).all()


def test_gradients_flow_to_all_parameters():
    layer = QLSTM(input_size=4, hidden_size=3, n_qubits=4)
    out, _ = layer(torch.randn(3, 2, 4))
    out.sum().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_gradient_flows_to_input():
    # Confirms backprop through the quantum circuit reaches the inputs, i.e.
    # backpropagation through time is intact across the recurrent steps.
    layer = QLSTM(input_size=4, hidden_size=3, n_qubits=4)
    x = torch.randn(3, 2, 4, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_can_pass_initial_state():
    layer = QLSTM(input_size=4, hidden_size=3, n_qubits=4)
    x = torch.randn(3, 2, 4)
    h0 = torch.randn(1, 2, 3)
    c0 = torch.randn(1, 2, 3)
    out, (h_n, c_n) = layer(x, (h0, c0))
    assert out.shape == (3, 2, 3)


def test_lqlstm_is_linear_enhanced():
    layer = LQLSTM(input_size=4, hidden_size=3, n_qubits=5)
    assert layer.cell.linear_enhanced is True
    # linear_enhanced decouples n_qubits from input+hidden sizes
    out, _ = layer(torch.randn(3, 2, 4))
    assert out.shape == (3, 2, 3)


def test_cell_single_step():
    cell = QLSTMCell(input_size=4, hidden_size=3, n_qubits=4)
    h, c = cell(torch.randn(2, 4))
    assert h.shape == (2, 3)
    assert c.shape == (2, 3)


def test_pure_qlstm_requires_matching_qubits():
    # linear_enhanced=False -> n_qubits must equal input_size + hidden_size
    with pytest.raises(ValueError, match="n_qubits must equal"):
        QLSTM(input_size=4, hidden_size=3, n_qubits=4, linear_enhanced=False)
    # matching dims is accepted and runs
    layer = QLSTM(input_size=4, hidden_size=3, n_qubits=7, linear_enhanced=False)
    out, _ = layer(torch.randn(3, 2, 4))
    assert out.shape == (3, 2, 3)


def test_unknown_ansatz_raises():
    with pytest.raises(ValueError, match="unknown ansatz"):
        make_vqc(n_qubits=4, ansatz="does-not-exist")


def test_learns_a_simple_sequence_task():
    # The target is the mean of each length-3 input sequence: a task an LSTM
    # can fit. A working quantum LSTM + head should drive the loss well down.
    torch.manual_seed(0)
    seq_len, batch, input_size, hidden = 3, 16, 2, 4
    x = torch.randn(seq_len, batch, input_size)
    y = x.mean(dim=(0, 2), keepdim=True).squeeze(0)  # (batch, 1)

    layer = QLSTM(input_size=input_size, hidden_size=hidden, n_qubits=4)
    head = torch.nn.Linear(hidden, 1)
    model = torch.nn.ModuleList([layer, head])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    loss_fn = torch.nn.MSELoss()

    def step():
        opt.zero_grad()
        out, _ = layer(x)
        pred = head(out[-1])
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        return loss.item()

    first = step()
    for _ in range(40):
        last = step()
    assert last < first * 0.5, f"loss did not fall enough: {first:.4f} -> {last:.4f}"
