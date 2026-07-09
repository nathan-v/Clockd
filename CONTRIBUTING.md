# Contributing to Clockd

## Code Style

This project uses `ruff` for linting and formatting. Please check your changes before submitting PRs:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Python Versions

Python 3.11+ is supported.

## Testing

### Unit Tests

Unit tests are written using pytest and should be included with every PR. If the PR is a bug fix, please include a regression test as well.

### Running Tests

```bash
pip install -e ".[dev,prometheus]"
pytest -v
```

### Code Coverage

Code coverage for this project is high and the intent is that it stays that way. PRs that reduce coverage should include justification.

```bash
coverage run -m pytest
coverage report --show-missing
```

## Developer Setup

```bash
git clone https://github.com/nathan-v/Clockd.git
cd Clockd
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,prometheus]"
pytest
```

## Pull Request Process

1. Ensure your code passes `ruff check` and `ruff format --check`
2. Add or update tests for your changes
3. Update documentation if your changes affect the API or configuration
4. Ensure all tests pass
5. Reference any related issues in your PR description
