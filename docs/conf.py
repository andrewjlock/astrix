# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add both the project root and src directories to the path
# This ensures sphinx can find the package both in dev mode and installed mode
# sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath(".."))
project = "AsTrIX"
copyright = "2025, Andrew Lock"
author = "Andrew Lock"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "myst_parser",
]

html_theme = "furo"

default_dark_mode = True
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": False,
    "member-order": "groupwise",  # Group by type (methods, properties, etc.)
}

autodoc_typehints = "description"
autodoc_member_order = "groupwise"
autosummary_generate = True  # generate stub pages automatically


# Configure MyST-Parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

suppress_warnings = ["myst.domains"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# add_module_names = False
html_domain_indices = False
html_compact_lists = False
#
# napoleon_use_ivar = True
# napoleon_use_rtype = False

add_module_names = False

autodoc_typehints = "description"
