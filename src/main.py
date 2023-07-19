import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as linalg
import math
import matplotlib.colors as mcolors

from plotter import Plotter

import sys, os

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))


class Lattice:
    def __init__(self, L: int, type: str) -> None:
        # str that store whether to use dense or spars matrices
        self.matrix_type = type
        self.L = L
        self.Hamiltonian_manual = np.zeros((L + 1, L + 1))
        self.Hamiltonian = np.zeros((2 ** (L + 1), 2 ** (L + 1)), dtype=np.complex128)

    def build_hamiltonian_manual(self, t: float, s: float) -> None:
        """
        builds the "handmade" Hamiltonian for a) in the h operator basis
        """
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
        """
        single site b operator
        """
        s1 = 0.5 * np.matrix([[0, 1], [1, 0]])
        s2 = 0.5 * np.matrix([[0, -1j], [1j, 0]])
        op = s1 + 1j * s2
        return op

    def expand_operator(self, op: np.matrix, j: int) -> np.matrix:
        """
        expansion func that kroneckers an operator at site j up to the full space
        """
        op = np.kron(np.eye(2**j, 2**j), op)
        op = np.kron(op, np.eye(2 ** (self.L - j), 2 ** (self.L - j)))

        return op

    def b_down_j(self, j: int) -> np.matrix:
        """
        builds full Hilbert space b operator on site j. use .getH() on this b to provide b^dagger
        """
        return self.expand_operator(self.b_down_single(), j)

    def correlator(self, j: int, l: int) -> np.matrix:
        """
        correlator for d)
        """
        return self.b_down_j(j).getH() @ self.b_down_j(l)

    def build_hamiltonian_ed(self, t: float, s: float) -> None:
        """
        buils Hamiltonian from b operators (which corresponds to S+ S- basis)
        """
        # clear Hamiltonian
        self.Hamiltonian = np.zeros(
            (2 ** (self.L + 1), 2 ** (self.L + 1)), dtype=np.complex128
        )

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
                self.b_down_j(j).getH() @ self.b_down_j(self.L)
                + self.b_down_j(j) @ self.b_down_j(self.L).getH()
            )

    def spectrum(self, type: str, rep: int, shareplot: bool = True) -> None:
        """
        provides plots for a) and c) depending on the "type" argument

        """
        # values of t and s for plot
        t = 1
        s_t = np.power(10, np.linspace(start=-2, stop=1, num=100))

        # manual for the a) plot
        if type == "manual":
            evals = np.ndarray((len(s_t), self.L + 1))

            for i in range(len(s_t)):
                self.build_hamiltonian_manual(t, s=s_t[i] * t)
                evals[i] = np.real(np.sort(np.linalg.eigvals(self.Hamiltonian_manual)))

        # exact for the c) plot
        elif type == "exact":
            # ndarray
            if self.matrix_type == "sparse":
                evals = []
                for i in range(len(s_t)):
                    self.build_hamiltonian_ed(t, s=s_t[i] * t)
                    self.Hamiltonian = sp.coo_matrix(self.Hamiltonian)

                    eigs, evectors = sp.linalg.eigsh(
                        self.Hamiltonian, which="SM", k=self.Hamiltonian.shape[0] - 2
                    )
                    evals.append(eigs)

                    self.Hamiltonian = self.Hamiltonian.toarray()
                evals = np.array(evals)
            # scipy.sparse
            elif self.matrix_type == "dense":
                evals = np.ndarray((len(s_t), 2 ** (self.L + 1)))
                for i in range(len(s_t)):
                    self.build_hamiltonian_ed(t, s=s_t[i] * t)
                    evals[i] = np.real(np.sort(np.linalg.eigvalsh(self.Hamiltonian)))
            else:
                raise Exception("matrix_type can either be `dense` or `sparse`")
        else:
            raise Exception("type can either be `manual` or `exact`")

        # reshape evals list according to the evolution of each eval under t_s
        evals_new = evals.T

        num_eigv = len(evals_new)
        # put all EV's on the same plot
        # Plot routine that colors the eigenvalues according to a specific coloring cycle
        # This allows to i.e. color all odd and even n Eigenvalues E_n in the same color or every third, fourth, ...
        if shareplot == True:
            P = Plotter(figsize=(6, 4), nrows=1, ncols=1, usetex=True)
            n = 0
            # rep assigns the number after which to cycle back to the starting color
            while n <= math.ceil(num_eigv / rep):
                # color list has something like 12 entries, should suffice
                colors = [
                    mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS
                ]
                for i in range(rep):
                    # first EV is black
                    if n * rep + i == 0:
                        P.ax.plot(
                            s_t,
                            evals_new[n * rep + i],
                            label=f"EV {n*rep+i}",
                            color="k",
                            lw=1.5,
                        )
                    # all intermediate EV's
                    elif n * rep + i < num_eigv - 1:
                        # plot with label for first round
                        if n * rep + i <= rep:
                            P.ax.plot(
                                s_t,
                                evals_new[n * rep + i],
                                label=f"EV {n*rep+i} +" + str(rep) + "n",
                                color=colors[i],
                                lw=0.7,
                            )
                        # plot without label for the repeats
                        else:
                            P.ax.plot(
                                s_t, evals_new[n * rep + i], color=colors[i], lw=0.7
                            )
                    # last EV is magenta
                    elif n * rep + i == num_eigv - 1:
                        P.ax.plot(
                            s_t,
                            evals_new[n * rep + i],
                            label="Last EV",
                            color="m",
                            lw=1.5,
                        )
                n += 1

            # make plot fancy
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

        # plot every EV on separate subplot (better with pdf)
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
            os.path.join(HOME_FOLDER, "..", "plots", "spectrum_" + str(type) + ".png")
        )

    def condensate_frac(self) -> None:
        """
        provides plot for d)
        """
        # stores initial L value
        L_init = self.L
        # prepare plot
        G = Plotter(figsize=(6, 12), nrows=2, ncols=1)
        rho_values = []
        # storage for the cond frac of this L
        n_0N_frac = []
        # Now loops over all L up to the L that the class was initiated with
        for L in range(1, L_init + 1):
            # adjust number of sites
            self.__init__(L, self.matrix_type)
            # values of t and s for plot
            t = 1
            s_t = np.power(10, np.linspace(start=-3, stop=1, num=200))
            # get pho for every values of s(t)
            for i in range(len(s_t)):
                rho = np.ndarray((self.L, self.L))
                self.build_hamiltonian_ed(t=t, s=s_t[i] * t)

                if self.matrix_type == "sparse":
                    self.Hamiltonian = sp.coo_matrix(self.Hamiltonian)
                    evalues, evectors = sp.linalg.eigsh(self.Hamiltonian)
                    self.Hamiltonian = self.Hamiltonian.toarray

                elif self.matrix_type == "dense":
                    evalues, evectors = np.linalg.eig(self.Hamiltonian)

                else:
                    raise Exception("matrix_type can either be `dense` or `sparse`")

                # get ground state of Hamiltonian
                ground = evectors[:, np.where(evalues == min(evalues))][:, :, 0]

                # fill rho matrix for this s(t)
                for j in range(self.L):
                    for l in range(self.L):
                        corr = self.correlator(j, l)
                        p = ground.conj().T @ corr @ ground
                        rho[j][l] = np.real(p[0, 0])

                # get rho evals
                evalues_rho = np.linalg.eigvals(rho)
                print(np.allclose(rho, np.diag(evalues_rho)))
                print(rho)
                print(evalues_rho)
                # get N
                N = np.round(np.trace(rho))
                # store and sort N/L = rho values to determine list of uniquely reoccuring rhos
                rho_values = np.append(rho_values, N / L)
                rho_values = np.sort(np.unique(rho_values))
                # store s_t, condensation fraction, rho, L
                N, L = np.real(N), np.real(L)
                n_0N_frac.append([s_t[i], max(evalues_rho) / N, N / L, L])
                # colormap for the rhos
                colors = [
                    mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS
                ]
        n_0N_frac = np.array(n_0N_frac)
        # keep track of the rhos already plotted with legend
        L_already_labeled = []
        # keep track of the rhos already plotted with legend
        rho_already_labeled = []
        # plot condensation frac against s_t and color according to rho
        for x in n_0N_frac:
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
                + str(self.matrix_type)
                + " matrices"
            )
            G.ax[i].set_xlabel(r"$s/t$")
            G.ax[i].set_ylabel(r"$\frac{n_0}{N}$")
            G.ax[i].legend()
        G.savefig(os.path.join(HOME_FOLDER, "..", "plots", "condensate_fraction.png"))
        # revert to original L just for safety
        self.__init__(L_init, self.matrix_type)


if __name__ == "__main__":
    L = 6
    # "dense" uses ndarray, "sparse" uses scipy.sparse.coo_matrix
    test = Lattice(L, "dense")

    # "manual" gives the a) spectrum, anything else gives the c) spectrum
    # the integer input determines after what number of EV's the color scheme repeats, i.e. 2 means that the even and odd EV's are colored alike

    test.spectrum("manual", 2, True)
    test.spectrum("exact", 2, True)
    # test.condensate_frac()

    def fock_to_product_state(statestring: list):
        zero = np.array([1, 0])
        one = np.array([0, 1])

        statevectors = [zero, one]

        state = statevectors[statestring[0]]
        L = len(statestring)
        for i in range(1, L):
            state = np.kron(state, statevectors[statestring[i]])

        return state

    print(np.real(test.b_down_j(0).getH() @ fock_to_product_state([0, 0, 0])))
    print(fock_to_product_state([1, 0, 0]))
