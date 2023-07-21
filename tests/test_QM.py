#!/usr/bin/env python3

import sys, os

base_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(base_dir, "..", "src"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir) 

import numpy as np
from main import Lattice
import QM

def test_fock_state():
    test = Lattice(L = 2, type_ = "sparse")

    init_state    = QM.HBFockState.from_fock_repr(vector = [0, 0, 0], typ = QM.ST.KET)
    target_states = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    for i, state in enumerate(target_states):
        tstate = QM.HBFockState.from_fock_repr(vector = state, typ = QM.ST.BRA)

        assert (tstate @ test.b_down_j(i).dagger() @ init_state) == 1

def test_norm():
    state = QM.HBFockState.from_fock_repr(vector = [1, 0, 1], typ = QM.ST.KET)
    assert np.isclose(state.dagger() @ state, 1)
