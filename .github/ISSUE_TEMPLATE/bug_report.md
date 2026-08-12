---
name: Bug report
about: Report a problem so it can be fixed
title: ""
labels: bug
assignees: ""
---

## What happened

A clear description of the problem.

## How to reproduce

A minimal script that shows the problem. For example:

```python
import torch
from qlstm import QLSTM

layer = QLSTM(input_size=8, hidden_size=4, n_qubits=4)
out, _ = layer(torch.randn(6, 2, 8))
# ...what goes wrong
```

## What you expected

What should have happened instead.

## Error output

The full traceback or wrong result, if any.

```
paste here
```

## Environment

- `qlstm` version:
- Python version:
- PyTorch version:
- PennyLane version:
- Operating system:
