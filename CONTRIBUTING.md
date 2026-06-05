# Contributing

## Development Setup

```bash
git clone https://github.com/TARVIA-lab/biomarker-discovery-pipeline.git
cd biomarker-discovery-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

## Making Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Test: `pytest tests/`
4. Commit: `git commit -am "Add feature description"`
5. Push: `git push origin feature/your-feature`
6. Create a pull request

## Code Style

- PEP 8 compliant
- Type hints encouraged
- Docstrings for all functions

## Testing

Run tests before submitting PRs:
```bash
pytest tests/ -v --cov=src
```

## Questions?

Open an issue or email o.lujano13@gmail.com
