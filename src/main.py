import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as linalg

from plotter import Plotter

import sys, os

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))


class Lattice:
    def __init__(self, L: int, type: str) -> None:
        self.matrix_type = type
        self.L = L
        self.Hamiltonian_manual = np.zeros((L + 1, L + 1))
        self.Hamiltonian = np.zeros((2 ** (L), 2 ** (L)), dtype=np.complex128)

    # Hamiltonian in h operator basis
    def build_hamiltonian_manual(self, t: float, s: float) -> None:
        # clear Hamiltonian
        self.Hamiltonian_manual = np.zeros((self.L + 1, self.L + 1))

        for i in range(self.L + 1):
            for j in range(self.L + 1):
                if i != j:
                    if i <= self.L - 1 and j <= self.L - 1:
                        self.Hamiltonian_manual[i][j] = -t
                    else:
                        self.Hamiltonian_manual[i][j] = -s

    def b_down_single(self) -> np.matrix:
        s1 = 0.5 * np.matrix([[0, 1], [1, 0]])
        s2 = 0.5 * np.matrix([[0, -1j], [1j, 0]])
        op = s1 + 1j * s2
        return op

    def expand_operator(self, op: np.matrix, j: int) -> np.matrix:
        op = np.kron(np.eye(2**j, 2**j), op)
        op = np.kron(op, np.eye(2 ** (self.L - j - 1), 2 ** (self.L - j - 1)))
        return op

    def b_down_j(self, j: int) -> np.matrix:
        return self.expand_operator(self.b_down_single(), j)

    def correlator(self, j: int, l: int) -> np.matrix:
        return self.b_down_j(j).getH() @ self.b_down_j(l)

    def build_hamiltonian_ed(self, t: float, s: float) -> None:
        # clear Hamiltonian
        self.Hamiltonian = np.zeros((2 ** (self.L), 2 ** (self.L)), dtype=np.complex128)

        for j in range(self.L - 1):
            # PBC
            if j == self.L - 2:
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() @ self.b_down_j(0)
                    + self.b_down_j(j) @ self.b_down_j(0).getH()
                )
            else:
                # print(j)
                # print(self.b_down_j(j + 1).shape)
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() @ self.b_down_j(j + 1)
                    + self.b_down_j(j) @ self.b_down_j(j + 1).getH()
                )
            # center hop part doesnt need any PBC separation
            self.Hamiltonian += -s * (
                self.b_down_j(j).getH() @ self.b_down_j(self.L - 1)
                + self.b_down_j(j) @ self.b_down_j(self.L - 1).getH()
            )

    def spectrum(self, type: str, stepsize: float, shareplot: bool = True) -> None:
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))

        # get eval for all ts combinations
        if type == "manual":
            evals = np.ndarray((len(s_t), self.L + 1))

            for i in range(len(s_t)):
                self.build_hamiltonian_manual(t, s=s_t[i] * t)
                evals[i] = np.real(np.sort(np.linalg.eigvals(self.Hamiltonian_manual)))

        else:
            if self.matrix_type == "sparse":
                evals = []
                for i in range(len(s_t)):
                    self.build_hamiltonian_ed(t, s=s_t[i] * t)
                    self.Hamiltonian = sp.coo_matrix(self.Hamiltonian)

                    eigs, evectors = sp.linalg.eigsh(self.Hamiltonian)
                    evals.append(np.real(np.sort(eigs)))

                    self.Hamiltonian = self.Hamiltonian.toarray()
                evals = np.array(evals)
            else:
                evals = np.ndarray((len(s_t), 2**self.L))
                for i in range(len(s_t)):
                    self.build_hamiltonian_ed(t, s=s_t[i] * t)
                    evals[i] = np.real(np.sort(np.linalg.eigvals(self.Hamiltonian)))

        # reshape evals list according to the evolution of each eval under t_s
        evals_new = evals.T

        num_eigv = len(evals_new)
        if shareplot == True:
            P = Plotter(figsize=(6, 4), nrows=1, ncols=1, usetex=True)
            for n in range(num_eigv):
                P.ax.plot(
                    s_t,
                    evals_new[n],
                    label=f"EV {n}",
                )
                P.ax.legend()
                P.ax.set_xscale("log")
                P.ax.set_title(
                    "Spectrum "
                    + str(type)
                    + ", "
                    + str(self.matrix_type)
                    + " matrices"
                    + f", System size: {self.L}"
                )
                P.ax.set_xlabel(r"$s/t$")
                P.ax.set_ylabel(r"$E_n$")
        else:
            P = Plotter(figsize=(6, 3 * num_eigv), nrows=num_eigv, ncols=1, usetex=True)
            for n in range(num_eigv):
                P.ax[n].plot(
                    s_t,
                    evals_new[n],
                    label=f"EV {n}",
                )
                P.ax[n].legend()
                P.ax[n].set_xscale("log")
                P.ax.set_title(
                    "Spectrum "
                    + str(type)
                    + ", "
                    + str(self.matrix_type)
                    + " matrices"
                    + f", System size: {self.L}"
                )
                P.ax.set_xlabel(r"$s/t$")
                P.ax.set_ylabel(r"$E_n$")

        P.savefig(
            os.path.join(HOME_FOLDER, "..", "plots", "spectrum_" + str(type) + ".pdf")
        )

    def condensate_frac(self) -> None:
        assert self.L >= 3, "L should be larger than 2 for reasonable result"
        L_init = self.L
        # Now loops over all L up to the L that the class was initiated with
        G = Plotter(figsize=(6, 4), nrows=1, ncols=1)
        for L in range(3, L_init + 1):
            self.__init__(L, self.matrix_type)
            n_0N_frac = []
            # values of t and s for plot
            t = 1
            s_t = np.power(10, np.linspace(start=-3, stop=1, num=200))
            for i in range(len(s_t)):
                rho = np.ndarray((self.L, self.L))
                self.build_hamiltonian_ed(t=t, s=s_t[i] * t)

                if self.matrix_type == "sparse":
                    self.Hamiltonian = sp.coo_matrix(self.Hamiltonian)
                    evalues, evectors = sp.linalg.eigsh(self.Hamiltonian)
                    self.Hamiltonian = self.Hamiltonian.toarray

                else:
                    evalues, evectors = np.linalg.eig(self.Hamiltonian)

                ground = evectors[:, np.where(evalues == min(evalues))][:, :, 0]
                for j in range(self.L):
                    for l in range(self.L):
                        p = ground.conj().T @ self.correlator(j, l) @ ground
                        rho[j][l] = np.real(p[0, 0])

                evalues_rho = np.linalg.eigvals(rho)
                n_0N_frac.append(max(evalues_rho) / np.trace(rho))
            G.ax.scatter(s_t, n_0N_frac, s=3, label=f"L = {L}")

        G.ax.set_xscale("log")
        G.ax.set_ylim(0, 1)
        G.ax.set_title(
            r"Condensate fraction $\frac{n_0}{N}$ with "
            + str(self.matrix_type)
            + " matrices"
        )
        G.ax.set_xlabel(r"$s/t$")
        G.ax.set_ylabel(r"$E_n$")
        G.ax.legend()
        G.savefig(os.path.join(HOME_FOLDER, "..", "plots", "condensate_fraction.pdf"))
        self.__init__(L_init, self.matrix_type)


if __name__ == "__main__":
    L = 8
    # "dense" uses ndarray, "sparse" uses scipy.sparse.coo_matrix
    test = Lattice(L, "dense")
    # "manual" gives the a) spectrum, anything else gives the c) spectrum
    test.spectrum("manual", 0.5, True)
    test.spectrum("exact", 1, True)
    test.condensate_frac()
