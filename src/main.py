#!/usr/bin/env python3

import numpy as np

import math
import matplotlib.colors as mcolors

from plotter import Plotter

import sys, os

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))

from QM import *


class Lattice:
    def __init__(self, L: int, type: str) -> None:
        # str that store whether to use dense or spars matrices
        self.matrix_type = type
        self.sparse = self.matrix_type == "sparse"

        self.L = L
        self.Hamiltonian_manual = Operator(
            L=L + 1, matrix=np.zeros((L + 1, L + 1)), sparse=self.sparse
        )
        self.Hamiltonian = HBFockOperator(
            L, sparse=self.sparse
        )  # Size 2**(L+1) x 2**(L+1)

    def build_hamiltonian_manual(self, t: float, s: float) -> None:
        """
        builds the "handmade" Hamiltonian for a) in the h operator basis
        """
        # clear Hamiltonian
        H = np.zeros((self.L + 1, self.L + 1))

        for i in range(self.L + 1):
            for j in range(self.L + 1):
                if i != j:
                    if i <= self.L - 1 and j <= self.L - 1:
                        H[i][j] = -t
                    else:
                        H[i][j] = -s

        self.Hamiltonian_manual = Operator(L=self.L + 1, matrix=H, sparse=False)

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
        self.Hamiltonian = HBFockOperator(self.L, sparse=self.sparse)

        for j in range(self.L):
            # PBC
            if j == self.L - 1:
                self.Hamiltonian += (
                    self.b_down_j(j).dagger() @ self.b_down_j(0)
                    + self.b_down_j(j) @ self.b_down_j(0).dagger()
                ) * -t
            else:
                self.Hamiltonian += (
                    self.b_down_j(j).dagger() @ self.b_down_j(j + 1)
                    + self.b_down_j(j) @ self.b_down_j(j + 1).dagger()
                ) * -t
            # center hop part doesnt need any PBC separation
            self.Hamiltonian += (
                self.b_down_j(j).dagger() @ self.b_down_j(self.L)
                + self.b_down_j(j) @ self.b_down_j(self.L).dagger()
            ) * -s

    def spectrum_manual(self) -> None:
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))

        evals = []

        for i in range(len(s_t)):
            self.build_hamiltonian_manual(t, s=s_t[i] * t)
            evals.append(self.Hamiltonian_manual.get_eigvals())

        evals_new = np.array(evals).T
        num_eigv = len(evals_new)

        P = Plotter(figsize=(6, 4), nrows=1, ncols=1, usetex=True)

        P.ax.plot(
            s_t,
            evals_new[0],
            label=f"EV {0}",
            color="k",
            lw=1.5,
        )
        P.ax.plot(
            s_t,
            evals_new[1],
            label=f"all other EV's",
            color="b",
            lw=0.7,
        )
        for i in range(2, num_eigv - 1):
            # plot without label for the repeats
            P.ax.plot(
                s_t,
                evals_new[i],
                color="b",
                lw=0.7,
            )

        # last EV is magenta
        P.ax.plot(
            s_t,
            evals_new[-1],
            label="Last EV",
            color="m",
            lw=1.5,
        )

        P.ax.legend()
        P.ax.set_xscale("log")
        P.ax.set_title(
            "Spectrum manual, "
            + str(self.matrix_type)
            + " matrices"
            + f", System size: {self.L}"
        )
        P.ax.set_xlabel(r"$s/t$")
        P.ax.set_ylabel(r"$E_n$")

        P.savefig(os.path.join(HOME_FOLDER, "..", "plots", "spectrum_manual.png"))

    def spectrum_exact(self) -> None:
        """
        provides plots for c)

        """
        rep = self.L
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))

        evals = []
        for i in range(len(s_t)):
            self.build_hamiltonian_ed(t, s=s_t[i] * t)
            evals.append(self.Hamiltonian.get_eigvals())

        # reshape evals list according to the evolution of each eval under t_s
        evals_new = np.array(evals).T
        num_eigv = len(evals_new)

        # colorlist
        colors = [mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS]

        assert rep - 1 <= len(colors), "Not enough unique colors"

        P = Plotter(figsize=(6, 4 * (rep - 1)), nrows=rep - 3, ncols=1, usetex=True)

        for rep_local in range(2, rep - 1):
            n = 0
            while n <= math.ceil(num_eigv / rep_local):
                for i in range(rep_local):
                    # first EV is black
                    if n * rep_local + i == 0:
                        P.ax[rep_local - 2].plot(
                            s_t,
                            evals_new[n * rep_local + i],
                            label=f"EV {n*rep_local+i}",
                            color="k",
                            lw=1.5,
                        )

                    # all intermediate EV's
                    elif n * rep_local + i < num_eigv - 1:
                        # plot with label for first round
                        if n * rep_local + i <= rep_local:
                            P.ax[rep_local - 2].plot(
                                s_t,
                                evals_new[n * rep_local + i],
                                label=f"EV {n*rep_local+i} +"
                                + str(rep_local)
                                + r"$n, n \in {1,2,...}$",
                                color=colors[i],
                                lw=0.7,
                            )

                        # plot without label for the repeats
                        else:
                            P.ax[rep_local - 2].plot(
                                s_t,
                                evals_new[n * rep_local + i],
                                color=colors[i],
                                lw=0.7,
                            )

                    # last EV is magenta
                    elif n * rep_local + i == num_eigv - 1:
                        P.ax[rep_local - 2].plot(
                            s_t,
                            evals_new[n * rep_local + i],
                            label="Last EV",
                            color="m",
                            lw=1.5,
                        )

                n += 1

            P.ax[rep_local - 2].legend()
            P.ax[rep_local - 2].set_xscale("log")
            P.ax[rep_local - 2].set_title(
                "Spectrum exact, "
                + str(self.matrix_type)
                + " matrices"
                + f", System size: {self.L}"
                + f", reps = {rep_local}"
            )
            P.ax[rep_local - 2].set_xlabel(r"$s/t$")
            P.ax[rep_local - 2].set_ylabel(r"$E_n$")

        P.savefig(os.path.join(HOME_FOLDER, "..", "plots", "spectrum_exact.png"))


def condensate_frac(init_L, matrix_type) -> None:
    """
    provides plot for d)
    """
    # stores initial L value
    L_init = init_L

    # prepare plot
    G = Plotter(figsize=(6, 12), nrows=2, ncols=1)
    # colorlist
    colors = [mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS]
    rho_values = []
    # storage for the cond frac of this L
    n_0N_frac = []
    # Now loops over all L up to the L that the class was initiated with
    for L in range(1, L_init + 1):
        # adjust number of sites
        lattice = Lattice(L, matrix_type)
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-3, stop=3, num=200))
        # get pho for every values of s(t)
        for i in range(len(s_t)):
            rho = np.ndarray((L + 1, L + 1))
            lattice.build_hamiltonian_ed(t=t, s=s_t[i] * t)

            evalues, evectors = lattice.Hamiltonian.get_eigsys()

            # get ground state of Hamiltonian
            ground = evectors[:, np.where(evalues == min(evalues))][:, :, 0]
            ground = HBFockState(L=L, vector=ground, typ=ST.KET)

            # fill rho matrix for this s(t)
            for j in range(L):
                for l in range(L):
                    corr = lattice.correlator(j, l)
                    p = ground.dagger() @ corr @ ground
                    rho[j][l] = np.real(p)

            rho = np.where(np.isnan(rho) == True, 0, rho)

            # get rho evals
            evalues_rho = np.linalg.eigvals(rho)
            # get N
            N = np.round(np.trace(rho))
            # store and sort N/L = rho values to determine list of uniquely reoccuring rhos
            rho_values = np.append(rho_values, N / L)
            rho_values = np.sort(np.unique(rho_values))
            # store s_t, condensation fraction, rho, L
            N, L = np.real(N), np.real(L)
            n_0N_frac.append([s_t[i], max(evalues_rho) / N, N / L, L])

    n_0N_frac = np.array(n_0N_frac)
    # keep track of the rhos already plotted with legend
    L_already_labeled = []
    # keep track of the rhos already plotted with legend
    rho_already_labeled = []
    # plot condensation frac against s_t and color according to rho
    for x in n_0N_frac:
        # check for nan's and infs (low L's may have 0 particles in them)
        if (
            np.any(np.isnan(x[1])) == True
            or np.any(np.isinf(x[1])) == True
            or int(x[2]) >= 1
            or np.real(x[1]) >= 1
        ):
            continue
        else:
            # odd L's
            if np.real(x[3]) % 2 != 0:
                # label with rho and L values and choose new color if rho is new
                if x[2] not in rho_already_labeled:
                    x = np.real(x)
                    G.ax[0].scatter(
                        x[0],
                        x[1],
                        s=3,
                        label=r"$\rho$ = %.3f" % (x[2]) + f", L = {int(x[3])}",
                        color=colors[np.where(rho_values == x[2])[0][0]],
                    )
                    rho_already_labeled.append(x[2])
                # plot unlabeled if rho already appeared
                else:
                    G.ax[0].scatter(
                        x[0],
                        x[1],
                        s=3,
                        color=colors[np.where(rho_values == x[2])[0][0]],
                    )
            # even L's
            else:
                # label with rho and L values and choose new color if rho is new
                if x[3] not in L_already_labeled:
                    x = np.real(x)
                    G.ax[1].scatter(
                        x[0],
                        x[1],
                        s=3,
                        label=r"$\rho$ = %.3f" % (x[2]) + f", L = {int(x[3])}",
                        color=colors[int(x[3])],
                    )
                    L_already_labeled.append(x[3])
                # plot unlabeled if rho already appeared
                else:
                    G.ax[1].scatter(
                        x[0],
                        x[1],
                        s=3,
                        color=colors[int(x[3])],
                    )

    # Plotting
    for i in range(2):
        G.ax[i].set_xscale("log")
        G.ax[i].set_ylim(0, 1)
        G.ax[i].set_title(
            r"Condensate fraction $\frac{n_0}{N}$ with "
            + str(matrix_type)
            + " matrices"
        )
        G.ax[i].set_xlabel(r"$s/t$")
        G.ax[i].set_ylabel(r"$\frac{n_0}{N}$")
        G.ax[i].legend()
    G.savefig(os.path.join(HOME_FOLDER, "..", "plots", "condensate_fraction.png"))


if __name__ == "__main__":
    L = 9
    # "dense" uses ndarray, "sparse" uses scipy.sparse.coo_matrix
    test = Lattice(L, "sparse")

    # "manual" gives the a) spectrum, exact else gives the c) spectrum
    # test.spectrum_manual()
    # test.spectrum_exact()

    condensate_frac(init_L=L, matrix_type="dense")
