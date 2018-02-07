# Allow printing Unicode characters
import os
os.environ["PYTHONIOENCODING"] = 'UTF-8'

# Execute locate.py script
from . import locate as plant_locator
