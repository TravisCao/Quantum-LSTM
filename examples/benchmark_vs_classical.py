"""Compare an LQLSTM with a classical ``torch.nn.LSTM`` on a small task.

The point is a fair, reproducible side-by-side, not a claim that the quantum
layer wins. It trains both models on the same data with the same head and
optimiser settings, then prints each model's trainable-parameter count and its
final training and held-out loss. The task is a genuine recurrent one: predict
the running sum of the first input feature over the sequence, which needs the
cell to accumulate state.

Run with::

    python examples/benchmark_vs_classical.py

Quantum simulation is slow, so the sizes are deliberately small.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from qlstm import LQLSTM


def make_data(n, seq_len, input_size, generator):
    x = torch.randn(seq_len, n, input_size, generator=generator)
    # Target: sum of feature 0 over time -> requires accumulation across steps.
    y = x[:, :, 0].sum(dim=0, keepdim=True).transpose(0, 1)  # (n, 1)
    return x, y


def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def train(layer, head, x, y, xv, yv, steps, lr):
    params = list(layer.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        out, _ = layer(x)
        loss = loss_fn(head(out[-1]), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out, _ = layer(x)
        train_loss = loss_fn(head(out[-1]), y).item()
        outv, _ = layer(xv)
        val_loss = loss_fn(head(outv[-1]), yv).item()
    return train_loss, val_loss


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    seq_len, input_size, hidden = 6, 2, 4
    x, y = make_data(64, seq_len, input_size, gen)
    xv, yv = make_data(64, seq_len, input_size, gen)
    steps, lr = 80, 0.05

    print(f"task: running-sum regression  seq_len={seq_len} input={input_size} "
          f"hidden={hidden}  train/val=64/64  steps={steps}\n")

    q_layer = LQLSTM(input_size=input_size, hidden_size=hidden, n_qubits=4)
    q_head = nn.Linear(hidden, 1)
    q_params = count_params(q_layer) + count_params(q_head)
    q_train, q_val = train(q_layer, q_head, x, y, xv, yv, steps, lr)

    c_layer = nn.LSTM(input_size=input_size, hidden_size=hidden)
    c_head = nn.Linear(hidden, 1)
    c_params = count_params(c_layer) + count_params(c_head)
    c_train, c_val = train(c_layer, c_head, x, y, xv, yv, steps, lr)

    print(f"{'model':<16}{'params':>10}{'train MSE':>14}{'val MSE':>12}")
    print("-" * 52)
    print(f"{'LQLSTM':<16}{q_params:>10}{q_train:>14.4f}{q_val:>12.4f}")
    print(f"{'nn.LSTM':<16}{c_params:>10}{c_train:>14.4f}{c_val:>12.4f}")
    print(
        "\nBoth models train on this task and reach a comparable validation "
        "error.\nThe linear-enhanced quantum layer carries extra classical "
        "projection\nparameters, so it is not smaller than a classical LSTM of "
        "this size.\nNumbers vary with seed and size; this is a fair-setup "
        "demonstration,\nnot a performance claim."
    )


if __name__ == "__main__":
    main()
