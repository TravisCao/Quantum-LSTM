"""Minimal quickstart: train an LQLSTM on a tiny synthetic sequence task.

Run with::

    python examples/quickstart.py

The task is to predict the mean of each short input sequence. The loss should
fall clearly within a few dozen steps, confirming the quantum layer trains.
"""

import torch

from qlstm import LQLSTM

torch.manual_seed(0)

seq_len, batch, input_size, hidden = 4, 16, 3, 4
x = torch.randn(seq_len, batch, input_size)  # (seq, batch, feature)
y = x.mean(dim=(0, 2), keepdim=True).squeeze(0)  # (batch, 1)

layer = LQLSTM(input_size=input_size, hidden_size=hidden, n_qubits=4)
head = torch.nn.Linear(hidden, 1)
params = list(layer.parameters()) + list(head.parameters())
opt = torch.optim.Adam(params, lr=0.05)
loss_fn = torch.nn.MSELoss()

for step in range(60):
    opt.zero_grad()
    output, (h_n, c_n) = layer(x)  # output: (seq, batch, hidden)
    pred = head(output[-1])  # use the last step
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    if step % 10 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

print(f"final loss {loss.item():.4f}")
