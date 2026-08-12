# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-08-13

First packaged release. The quantum LSTM from the paper is now an installable,
tested PyTorch library (`pip install qlstm`) alongside the original
paper-reproduction code under `src/`.

### Added
- `QLSTM` and `LQLSTM` layers with a `torch.nn.LSTM`-compatible call signature
  (`(seq, batch, feature)` or `batch_first`, returning `output, (h_n, c_n)`).
- `QLSTMCell` for single-step use, with a `linear_enhanced` flag selecting the
  linear-enhanced L-QLSTM (default) or the classic QLSTM.
- `make_vqc`, a variational-circuit factory built from PennyLane templates
  (`basic` and `strong` ansätze; `X`/`Y`/`Z` angle embedding).
- Test suite covering output shapes, gradient flow to every parameter and to the
  inputs, a learning check, and the API surface.
- MIT license, packaging metadata, `CITATION.cff`, and CI on Python 3.10-3.12.
