import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(os.path.dirname(SCRIPT_DIR))