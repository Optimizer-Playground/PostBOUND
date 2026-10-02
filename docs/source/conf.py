# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

import postbound as pb

current_year = date.today().year
project = "PostBOUND"
copyright = f"2023-{current_year}, Dresden Database Research Group"
author = "Dresden Database Research Group"
release = pb.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "IPython.sphinxext.ipython_console_highlighting",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_favicon = "_static/favicon.ico"
html_logo = "../figures/postbound-logo.svg"
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "logo": {
        "text": f"PostBOUND {pb.__version__}",
        "height": "150px",
    }
}

# -- Options for API documentation -------------------------------------------------

autodoc_member_order = "bysource"

autodoc_typehints = "both"
