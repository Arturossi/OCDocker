import os
import sys
import logging
from unittest.mock import MagicMock

logging.basicConfig(level=logging.DEBUG)

# Insert the project root into the system path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, '/data/hd4tb/OCDocker/OCDocker')

# Mock modules
class Mock(MagicMock):
    @classmethod
    def __getattr__(cls, name):
        return MagicMock()

# Function to get all modules from a directory
def get_all_modules(directory):
    modules = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                module = os.path.relpath(os.path.join(root, file), directory)
                module = module[:-3].replace(os.path.sep, '.')
                modules.append(module)
    return modules

# Get all modules in the project
project_root = os.path.abspath('../../OCDocker')
all_modules = get_all_modules(project_root)

# Mock all modules
for module in all_modules:
    sys.modules[module] = Mock()

# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OCDocker'
copyright = '2024, Artur Duque Rossi'
author = 'Artur Duque Rossi'

version = '0.9.1'
release = '0.9.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinxarg.ext',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser'
]

autosummary_generate = True  # Turn on sphinx.ext.autosummary

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Napoleon settings -------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Autodoc settings --------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# -- Autodoc mock imports ----------------------------------------------------
autodoc_mock_imports = []
