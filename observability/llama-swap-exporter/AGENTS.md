# llama-swap-exporter

Language: python 3.14

## Code Standards

- Use modern python features and types
- Use `pytest` for testing
- Use `ruff` for linting and formatting
- Use `uv` for dependency management
- Use `pre-commit` for linting

## Testing

- Always run tests as you go
- Always run `pre-commit -a`
- Use fixtures for common logic and mocks in tests
- Put common fixtures needed in multiple tests files in `conftest.py`
- Put empty line before first assert call in tests
- Do not use `unittest.mock`, use `pytest-mock` instead
- Do not use import inside functions in tests
