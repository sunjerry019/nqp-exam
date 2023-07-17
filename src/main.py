import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as linalg

from plotter import Plotter

import sys, os

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))


class Lattice:
    def __init__(self, L: int) -> None:
        self.L = L
        self.Hamiltonian_manual = np.zeros((L + 1, L + 1))
        self.Hamiltonian = np.zeros((2 ** (L), 2 ** (L)), dtype=np.complex128)

    # Hamiltonian in h operator basis
    def build_hamiltonian_manual(self, t: float, s: float) -> None:
        for i in range(self.L + 1):
            for j in range(self.L + 1):
                if i != j:
                    if i <= self.L - 1 and j <= self.L - 1:
                        self.Hamiltonian_manual[i][j] = -t
                    else:
                        self.Hamiltonian_manual[i][j] = -s

    def b_down_j(self, j: int):
        assert j <= self.L - 1, "J too big"

        s1 = 0.5 * np.matrix([[0, 1], [1, 0]])
        s2 = 0.5 * np.matrix([[0, -1j], [1j, 0]])
        b = s1 + 1j * s2
        b_final = b
        for k in range(0, j):
            b_final = np.kron(np.eye(2), b_final)
        for k in range(j + 1, self.L):
            b_final = np.kron(b_final, np.eye(2))
        return b_final

    def correlator(self, j: int, l: int):
        return self.b_down_j(j).getH() @ self.b_down_j(l)

    def build_hamiltonian_ed(self, t: float, s: float) -> None:
        for j in range(self.L - 1):
            # PBC
            if j == self.L - 2:
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() @ self.b_down_j(0)
                    + self.b_down_j(j) @ self.b_down_j(0).getH()
                )
            else:
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() @ self.b_down_j(j + 1)
                    + self.b_down_j(j) @ self.b_down_j(j + 1).getH()
                )
            # center hop part doesnt need any PBC separation
            self.Hamiltonian += -s * (
                self.b_down_j(j).getH() @ self.b_down_j(self.L - 1)
                + self.b_down_j(j) @ self.b_down_j(self.L - 1).getH()
            )

    def spectrum(self, type: str, stepsize: float, shareplot: bool = True):
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))

        # get eval for all ts combinations
        if type == "manual":
            evals = np.ndarray((len(s_t), self.L + 1))
            for i in range(len(s_t)):
                self.build_hamiltonian_manual(t, s=s_t[i] * t)
                evals[i] = np.sort(np.linalg.eigvals(test.Hamiltonian_manual))
        else:
            evals = np.ndarray((len(s_t), 2**self.L))
            for i in range(len(s_t)):
                self.build_hamiltonian_ed(t, s=s_t[i] * t)
                evals[i] = np.sort(np.linalg.eigvals(test.Hamiltonian))

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

        P.savefig(
            os.path.join(HOME_FOLDER, "..", "plots", "spectrum_" + str(type) + ".pdf")
        )

    def condensate_frac(self):
        n_0N_frac = []
        # values of t and s for plot
        t = 1
        s_t = np.linspace(start=0, stop=10, num=100)
        rho = np.ndarray((self.L, self.L))
        for i in range(len(s_t)):
            self.build_hamiltonian_ed(t=t, s=s_t[i] * t)
            evalues, evectors = np.linalg.eig(test.Hamiltonian)
            ground = evectors[:, np.where(evalues == min(evalues))][:, :, 0]
            for j in range(self.L):
                for l in range(self.L):
                    p = ground.conj().T @ self.correlator(j, l) @ ground
                    rho[j][l] = np.real(p[0, 0])

            evalues = np.linalg.eigvals(rho)
            n_0N_frac.append(max(evalues) / np.trace(rho))

        P = Plotter(figsize=(6, 4), nrows=1, ncols=1)
        P.ax.scatter(s_t, n_0N_frac, s=3)
        P.savefig(os.path.join(HOME_FOLDER, "..", "plots", "condensate_fraction.pdf"))


if __name__ == "__main__":
    test = Lattice(4)
    #'manual' gives the a) spectrum, anything else gives the c) spectrum
    test.spectrum("manual", 0.5, True)
    test.spectrum("exact", 1, True)
    test.condensate_frac()
