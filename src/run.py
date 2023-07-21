#!/usr/bin/env python3

from main import Lattice, condensate_frac
from QM import *

import numpy as np
import sys

"""L = 2

while True:
    lattice = Lattice(L, "sparse", mpi = True)
    lattice.spectrum("manual")
    lattice.spectrum("exact")

    condensate_frac(L, matrix_type="dense", mpi = True)

    L += 1"""

for L in range(2,12):
    lattice = Lattice(L, "sparse", mpi = True)
    lattice.spectrum("manual")
    lattice.spectrum("exact")
