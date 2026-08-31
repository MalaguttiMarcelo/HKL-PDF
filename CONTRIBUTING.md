# Contributing

## Reporting bugs

Open a GitHub issue and include:

- Operating system
- Python version
- Package version or Git commit
- Complete traceback
- Minimal input file
- Minimal structure and data files, when redistribution is permitted
- Expected behavior
- Actual behavior

Do not include confidential experimental data.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -v