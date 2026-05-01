# torchff Documentation

This directory contains the Sphinx documentation for the torchff project.

## Building the Documentation

### Prerequisites

Install doc-build dependencies:

```bash
pip install sphinx sphinx-material myst-parser
```

### Build Commands

To build the HTML documentation:

```bash
cd docs
make html
```

The generated documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser to view it.

### Other Build Options

- `make clean` - Remove all build files
- `make html` - Build HTML documentation
- `make latexpdf` - Build PDF documentation (requires LaTeX)
- `make help` - Show all available build options
