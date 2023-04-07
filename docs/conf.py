# docs/conf.py
"""Sphinx configuration."""
project = "gym-distractions"
author = "Sebastian Markgraf"
copyright = f"2023, {author}"

# Add napoleon to the extensions list
extensions = ["sphinxcontrib.apidoc", "sphinx.ext.napoleon", "sphinx_autodoc_typehints"]

apidoc_module_dir = "../gym_distractions"
