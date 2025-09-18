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
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinx_gallery"
    # "sphinx_rtd_dark_mode"
]


html_theme = "sphinx_rtd_theme"
default_dark_mode = True
html_theme_options = {
    # "style_nav_header_background": "#2980B9",
    # "dark_mode_theme": "dark",  # Enable dark mode
    # ""
    # "preference_switch_enabled": True  # Show toggle switch
}


autosummary_generate = True              # generate stub pages automatically


# Configure MyST-Parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

suppress_warnings = ["myst.domains"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# add_module_names = False
# html_domain_indices = False
# html_compact_lists = True
#
# napoleon_use_ivar = True
# napoleon_use_rtype = False

# add_module_names = False

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    # "show-inheritance": True,
}
autodoc_typehints = "description"
