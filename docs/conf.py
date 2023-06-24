import sys
import os
project = "TRILL"
html_title = 'TRILL'
author = "Zachary A. Martinez"
sys.path.insert(0, os.path.abspath('..'))
extensions = ['sphinxarg.ext', 'myst_parser', 'sphinx_rtd_dark_mode', 'sphinx_copybutton']
autodoc_mock_imports = ["trill-proteins", "esm", "numpy", "pandas", "torch", "pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]
default_dark_mode = True
