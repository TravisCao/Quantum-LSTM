# Contributing to qlstm

Thanks for your interest in `qlstm`. Bug reports, feature requests, and pull
requests are all welcome.

## Ways to help

- **Report a bug.** Open an issue with the bug-report template. A short script
  that reproduces the problem is the fastest way to a fix.
- **Request a feature.** Open an issue with the feature-request template and
  describe the use case.
- **Send a pull request.** Fix a bug, add a test, improve the docs, or add a
  feature. For a large change, open an issue first so the design can be agreed
  before you write the code.

## Development setup

Python 3.10 or newer is required. Install the package and its test dependencies
in editable mode:

```bash
git clone https://github.com/TravisCao/Quantum-LSTM.git
cd Quantum-LSTM
pip install -e ".[test]"
```

This installs PyTorch, PennyLane, and pytest.

## Run the tests

```bash
pytest -q
```

The suite must pass before a pull request can merge. Continuous integration runs
the same command on Python 3.10 through 3.13, so run it locally first.

## Pull request checklist

1. Add or update a test for the behaviour you change. New features need a test
   that fails before the change and passes after it.
2. Keep the public API compatible with `torch.nn.LSTM` where the two overlap. A
   change that breaks that compatibility needs a clear reason in the pull
   request description.
3. Run `pytest -q` and confirm it is green.
4. Update `README.md` and `CHANGELOG.md` if you change behaviour, add a feature,
   or change the public interface.
5. Match the style of the surrounding code: type hints on public functions,
   docstrings on public classes and methods, and clear names.

## Scope

The installable `qlstm` package under [`qlstm/`](qlstm/) targets current PyTorch
and PennyLane. The original paper-reproduction code under [`src/`](src/) targets
the pinned versions in `requirement.txt` and is kept for reference; new library
work belongs in `qlstm/`.

## Questions

Open an [issue](https://github.com/TravisCao/Quantum-LSTM/issues) or contact
travisyjcao@gmail.com.
