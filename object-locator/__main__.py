# Copyright &copyright 2018 The Board of Trustees of Purdue University.
# All rights reserved.
# 
# This source code is not to be distributed or modified
# without the written permission of Edward J. Delp at Purdue University
# Contact information: ace@ecn.purdue.edu
# =====================================================================

# Allow printing Unicode characters
import os
os.environ["PYTHONIOENCODING"] = 'UTF-8'

# Execute locate.py script
from . import locate as object_locator
