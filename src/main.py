#!/usr/bin/env python3

import numpy as np

import math
import matplotlib.colors as mcolors

from plotter import Plotter

import sys, os

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))

from QM import *

try:
    from mpi4py import MPI
except ModuleNotFoundError as e:
    MPI = None


class Lattice:
    def __init__(self, L: int, type_: str, mpi: bool = False) -> None:
        # str that store whether to use dense or spars matrices
        self.matrix_type = type_
        self.sparse = self.matrix_type == "sparse"

        self.mpi = mpi if MPI is not None else False

        if self.mpi:
            self.MPI_COMM = MPI.COMM_WORLD
            self.MPI_SIZE = self.MPI_COMM.Get_size()
            self.MPI_RANK = self.MPI_COMM.Get_rank()

        self.L = L
        self.hamiltonian = {
            "manual": Operator(
                L=L + 1, matrix=np.zeros((L + 1, L + 1)), sparse=self.sparse
            ),
            "exact": HBFockOperator(L, sparse=self.sparse),  # Size 2**(L+1) x 2**(L+1)
        }

    def build_hamiltonian_manual(self, t: float, s: float) -> None:
        """
        builds the "handmade" Hamiltonian for a) in the h operator basis
        """
        # clear Hamiltonian
        H = np.zeros((self.L + 1, self.L + 1))

        for i in range(self.L + 1):
            for j in range(self.L + 1):
                if i == j + 1 or i == j - 1:
                    if i <= self.L - 1 and j <= self.L - 1:
                        H[i][j] = -t
                elif i == self.L or j == self.L: 
                    H[i][j] = -s

        self.hamiltonian["manual"] = Operator(L=self.L + 1, matrix=H, sparse=False)

    def b_down_single(self) -> HBFockOperator:
        """
        single site b operator
        """
        mat = np.array([[0, 1], [0, 0]])
        return HBFockOperator(0, mat, sparse=self.sparse)

    def b_down_j(self, j: int) -> HBFockOperator:
        """
        builds full Hilbert space b operator on site j. use .dagger() on this b to provide b^dagger
        """
        return self.b_down_single().expand_to(self.L, site=j)

    def correlator(self, j: int, l: int) -> HBFockOperator:
        """
        correlator for d)
        """
        corr = self.b_down_j(j).dagger() @ self.b_down_j(l)
        assert isinstance(corr, HBFockOperator)

        return corr

    def build_hamiltonian_ed(self, t: float, s: float) -> None:
        """
        buils Hamiltonian from b operators (which corresponds to S+ S- basis)
        """
        # clear Hamiltonian
        self.hamiltonian["exact"] = HBFockOperator(self.L, sparse=self.sparse)

        for j in range(self.L):
            # PBC
            if j == self.L - 1:
                self.hamiltonian["exact"] += (
                    self.b_down_j(j).dagger() @ self.b_down_j(0)
                    + self.b_down_j(j) @ self.b_down_j(0).dagger()
                ) * -t
            else:
                self.hamiltonian["exact"] += (
                    self.b_down_j(j).dagger() @ self.b_down_j(j + 1)
                    + self.b_down_j(j) @ self.b_down_j(j + 1).dagger()
                ) * -t
            # center hop part doesnt need any PBC separation
            self.hamiltonian["exact"] += (
                self.b_down_j(j).dagger() @ self.b_down_j(self.L)
                + self.b_down_j(j) @ self.b_down_j(self.L).dagger()
            ) * -s

    def spectrum(self, type_: str) -> None:
        """
        provides plots for a) and c) depending on the "type" argument

        """
        # values of t and s for plot

        build_hamiltonian_func = {
            "manual": self.build_hamiltonian_manual,  # manual for the a) plot
            "exact": self.build_hamiltonian_ed,  # exact for the c) plot
        }

        if type_ not in build_hamiltonian_func:
            raise ValueError("type can either be `manual` or `exact`")

        # We calculate the evals here
        if not self.mpi:
            t = 1
            s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))
            
            evals = []
            for i in range(len(s_t)):
                build_hamiltonian_func[type_](t, s=s_t[i] * t)
                evals.append(self.hamiltonian[type_].get_eigvals())
            evals = np.array(evals)
        else:
            # Buffers
            evals = []
            sendbuf = None
            recvbuf = None

            # Parameters
            t = 1
            num = 112  # num must be a multiple of MPI_SIZE
            chunk_size = num // self.MPI_SIZE
            assert (
                num % self.MPI_SIZE == 0
            ), "number of samples not divisible by number of nodes"

            # Just in case we have some other function running
            self.MPI_COMM.Barrier()

            # Generate the list of s/ts and scatter it
            if self.MPI_RANK == 0:
                s_t = np.power(10, np.linspace(start=-2, stop=1, num=num))
                sendbuf = np.array(np.split(s_t, self.MPI_SIZE))
            recvbuf = np.empty(chunk_size)
            self.MPI_COMM.Scatter(sendbuf, recvbuf, root=0)

            # Do the calculations
            _my_s_t = recvbuf
            for i in range(len(_my_s_t)):
                build_hamiltonian_func[type_](t, s=_my_s_t[i] * t)
                evals.append(self.hamiltonian[type_].get_eigvals())
            self.MPI_COMM.Barrier()

            # Gather back the evals
            sendbuf = np.array(evals)
            recvbuf = None
            if self.MPI_RANK == 0:
                recvbuf = np.empty([self.MPI_SIZE] + list(sendbuf.shape))
            self.MPI_COMM.Gather(sendbuf, recvbuf, root=0)

            if self.MPI_RANK != 0:
                return

            assert isinstance(recvbuf, np.ndarray)
            _shape = recvbuf.shape
            evals = recvbuf.reshape(_shape[0] * _shape[1], *_shape[2:])

        # From here on only rank zero if MPI

        # reshape evals list according to the evolution of each eval under t_s
        evals_new = evals.T

        if self.L > 9:
            np.savetxt(X = evals_new, fname = os.path.join('..','data',f'evals_new_{self.L}_{type_}.csv'), delimiter = ',')
            np.savetxt(X = s_t, fname = os.path.join('..','data',f's_t_{self.L}_{type_}.csv'), delimiter = ',')
        else:
            np.savetxt(X = evals_new, fname = os.path.join('..','data',f'evals_new_0{self.L}_{type_}.csv'), delimiter = ',')
            np.savetxt(X = s_t, fname = os.path.join('..','data',f's_t_0{self.L}_{type_}.csv'), delimiter = ',')

def condensate_frac(L, matrix_type, mpi: bool = False, writefile: bool = True) -> None:
    """
    provides plot for d)
    """

    datafile = os.path.join("..", "data", "CF", f"{L:02}.csv")

    if mpi:
        COMM = MPI.COMM_WORLD
        SIZE = COMM.Get_size()
        RANK = COMM.Get_rank()

    # adjust number of sites
    lattice = Lattice(L, matrix_type)
    if not mpi:
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-3, stop=3, num=200))
        # get pho for every values of s(t)

        for i in range(len(s_t)):
            rho = np.ndarray((L + 1, L + 1))
            lattice.build_hamiltonian_ed(t=t, s=s_t[i] * t)

            evalues, evectors = lattice.hamiltonian["exact"].get_eigsys()

            # get ground state of Hamiltonian
            ground = evectors[:, np.where(evalues == np.min(evalues))][:, :, 0]
            ground = HBFockState(L=L, vector=ground, typ=ST.KET)

            # fill rho matrix for this s(t)
            for j in range(L + 1):
                for l in range(L + 1):
                    corr = lattice.correlator(j, l)
                    p = ground.dagger() @ corr @ ground
                    rho[j][l] = np.real(p)

            # get rho evals
            evalues_rho = np.linalg.eigvals(rho)
            # get N
            N = np.trace(rho)
            # store s_t, condensation fraction, rho, L
            N, L = np.real(N), np.real(L)
            
            with open(datafile, 'a') as df:
                df.write(f"{s_t[i]},{np.max(evalues_rho) / N}, {N/(L+1)}\n")
    else:
        # Buffers
        evals = []
        sendbuf = None
        recvbuf = None

        # Parameters
        t = 1
        num = 256  # num must be a multiple of MPI_SIZE
        chunk_size = num // SIZE
        assert (
            num % SIZE == 0
        ), "number of samples not divisible by number of nodes"

        # Just in case we have some other function running
        COMM.Barrier()

        # Generate the list of s/ts and scatter it
        if RANK == 0:
            s_t = np.power(10, np.linspace(start=-3, stop=3, num=num))
            sendbuf = np.array(np.split(s_t, SIZE))
        recvbuf = np.empty(chunk_size)
        COMM.Scatter(sendbuf, recvbuf, root=0)

        # Do the calculations
        _my_s_t = recvbuf
        s_t_value = np.zeros(shape = (len(_my_s_t),))
        cond_frac = np.zeros(shape = (len(_my_s_t),))
        density   = np.zeros(shape = (len(_my_s_t),))

        for i in range(len(_my_s_t)):
            rho = np.ndarray((L + 1, L + 1))
            lattice.build_hamiltonian_ed(t=t, s=_my_s_t[i] * t)

            evalues, evectors = lattice.hamiltonian["exact"].get_eigsys()

            # get ground state of Hamiltonian
            ground = evectors[:, np.where(evalues == np.min(evalues))][:, :, 0]
            ground = HBFockState(L=L, vector=ground, typ=ST.KET)

            # fill rho matrix for this s(t)
            for j in range(L+1):
                for l in range(L+1):
                    corr = lattice.correlator(j, l)
                    p = ground.dagger() @ corr @ ground
                    rho[j][l] = np.real(p)


            # get rho evals
            evalues_rho = np.linalg.eigvals(rho)
            # get N
            N = np.trace(rho)
            # store s_t, condensation fraction, rho, L
            N, L = np.real(N), np.real(L)

            s_t_value[i] = _my_s_t[i]
            cond_frac[i] = np.real(np.max(evalues_rho)) / N
            density[i] = N/(L + 1)
        COMM.Barrier()

        # Gather back the evals
        recvbuf_s_t_value = None
        recvbuf_cond_frac = None
        recvbuf_density   = None
        if RANK == 0:
            recvbuf_s_t_value = np.empty([SIZE] + list(s_t_value.shape))
            recvbuf_cond_frac = np.empty([SIZE] + list(cond_frac.shape))
            recvbuf_density   = np.empty([SIZE] + list(density.shape))

        COMM.Gather(s_t_value, recvbuf_s_t_value, root=0)
        COMM.Gather(cond_frac, recvbuf_cond_frac, root=0)
        COMM.Gather(density, recvbuf_density, root=0)

        if RANK != 0:
            return
        
        assert RANK == 0
        # Here Rank == 0
        assert isinstance(recvbuf_s_t_value, np.ndarray)
        assert isinstance(recvbuf_cond_frac, np.ndarray)
        assert isinstance(recvbuf_density, np.ndarray)
        _shape_s_t_value = recvbuf_s_t_value.shape
        _shape_cond_frac = recvbuf_cond_frac.shape
        _shape_density   = recvbuf_density.shape

        s_t_value = recvbuf_s_t_value.reshape(_shape_s_t_value[0] * _shape_s_t_value[1], *_shape_s_t_value[2:])
        cond_frac = recvbuf_cond_frac.reshape(_shape_cond_frac[0] * _shape_cond_frac[1], *_shape_cond_frac[2:])
        density   = recvbuf_density.reshape(_shape_density[0] * _shape_density[1], *_shape_density[2:])

        if writefile:
            with open(datafile, 'a') as df:
                for k in range(len(s_t_value)):
                    df.write(f"{s_t_value[k]},{cond_frac[k]},{density[k]}\n")