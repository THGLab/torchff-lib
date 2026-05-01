"""Sphinx configuration file for torchff API documentation."""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# -- Project information -----------------------------------------------------

project = "torchff"
copyright = "2024, Eric Wang"
author = "Eric Wang"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
]

# Mock CUDA extension modules so autodoc works without compiling them
autodoc_mock_imports = [
    "torchff_bond",
    "torchff_angle",
    "torchff_torsion",
    "torchff_vdw",
    "torchff_dispersion",
    "torchff_slater",
    "torchff_coulomb",
    "torchff_multipoles",
    "torchff_amoeba",
    "torchff_ewald",
    "torchff_pme",
    "torchff_cmm",
    "torchff_nb",
    "torchff_nblist",
]

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_inherit_docstrings = False
autodoc_member_order = "bysource"

# Autosummary settings
autosummary_generate = True

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Templates
templates_path = ["_templates"]

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

# -- HTML output options -----------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "repository_url": "https://github.com/THGLab/torchff-lib",
    "use_repository_button": True,
    "use_issues_button": True,
    "home_page_in_toc": True,
}

html_title = "torchff"
html_static_path = ["_static"]

# -- Intersphinx mapping -----------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
