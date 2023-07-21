#!/usr/bin/env python3

# 

import sys, os

base_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, "..", "src"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir) 

import numpy as np
from main import *
import QM

def test_incompatibilities():
    L = 2
    lattice = Lattice(L = L, type_ = "dense", mpi = False)
    lattice.build_hamiltonian_ed(t = 1, s = 1)
    eigv = lattice.hamiltonian["exact"].get_eigvals()

    lattice = Lattice(L = L, type_ = "sparse", mpi = False)
    lattice.build_hamiltonian_ed(t = 1, s = 1)
    eigv = lattice.hamiltonian["exact"].get_eigvals()

if __name__ == "__main__":
    test_incompatibilities()
