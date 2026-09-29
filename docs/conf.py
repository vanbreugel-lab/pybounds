import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pybounds'
copyright = '2025, Ben Cellini, Burak Boyacioglu, Floris van Breugel'
author = 'Ben Cellini, Burak Boyacioglu, Floris van Breugel'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

html_theme = 'furo'
autodoc_member_order = 'bysource'
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Mock JAX so autodoc can render JAX class docstrings without JAX installed
autodoc_mock_imports = ['jax', 'jaxlib']

# Suppress warnings for relative links to repo files (notebooks, LICENSE) that
# are valid on GitHub but not resolvable within the Sphinx docs tree
suppress_warnings = ['myst.xref_missing']
