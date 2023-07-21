from main import Lattice, condensate_frac
from QM import *

import numpy as np
import sys

for L in range(2,11):
    condensate_frac(L, matrix_type="dense", mpi = True)
