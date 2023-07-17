import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.linalg as linalg


class lattice:
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
        return self.b_down_j(j).getH() * self.b_down_j(l)

    def build_hamiltonian_ed(self, t: float, s: float) -> None:
        for j in range(self.L - 1):
            # PBC
            if j == self.L - 1:
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() * self.b_down_j(0)
                    + self.b_down_j(j) * self.b_down_j(0).getH()
                )
            else:
                self.Hamiltonian += -t * (
                    self.b_down_j(j).getH() * self.b_down_j(j + 1)
                    + self.b_down_j(j) * self.b_down_j(j + 1).getH()
                )
            # center hop part doesnt need any PBC separation
            self.Hamiltonian += -s * (
                self.b_down_j(j).getH() * self.b_down_j(self.L - 1)
                + self.b_down_j(j) * self.b_down_j(self.L - 1).getH()
            )

    def spectrum(self, type: str, stepsize: float):
        # values of t and s for plot
        t, s = np.mgrid[0.1:20:stepsize, 0.1:20:stepsize]
        ts = np.vstack((t.flatten(), s.flatten())).T
        # storage for evals

        # get eval for all ts combinations

        if type == "manual":
            evals = np.ndarray((ts.shape[0], self.L + 1))
            for i in range(ts.shape[0]):
                self.build_hamiltonian_manual(ts[i][0], ts[i][1])
                evals[i] = np.linalg.eigvals(test.Hamiltonian_manual)
        else:
            evals = np.ndarray((ts.shape[0], 2**self.L))
            for i in range(ts.shape[0]):
                self.build_hamiltonian_ed(ts[i][0], ts[i][1])
                evals[i] = np.linalg.eigvals(test.Hamiltonian)

        # compute s/t fraction for all ts combinations
        ts_frac = [ts[i][1] / ts[i][0] for i in range(ts.shape[0])]

        # reshape evals list according to the evolution of each eval under ts_frac
        evals_new = np.array([np.take(evals, n, 1) for n in range(self.L + 1)])

        # plot for all evals
        for n in range(len(evals_new)):
            evals_sort = [y for x, y in sorted(zip(ts_frac, evals_new[n]))]
            ts_sort = [x for x, y in sorted(zip(ts_frac, evals_new[n]))]
            plt.plot(
                ts_sort,
                evals_sort,
                label="EV %.0f" % n,
            )
        plt.legend()
        plt.show()
        # plt.savefig('/spectrum_'+str(type)+'.png')

    def condensate_frac(self):
        n_0N_frac = []
        # values of t and s for plot
        t, s = np.mgrid[0.1:20:1, 0.1:20:1]
        ts = np.vstack((t.flatten(), s.flatten())).T
        # storage for evals
        rho = np.ndarray((self.L, self.L))
        for i in range(ts.shape[0]):
            self.build_hamiltonian_ed(ts[i][0], ts[i][1])
            evalues, evectors = np.linalg.eig(test.Hamiltonian)
            ground = evectors[:, np.where(evalues == min(evalues))][:, :, 0]
            for j in range(self.L):
                for l in range(self.L):
                    p = ground.conj().T * self.correlator(j, l) * ground
                    rho[j][l] = np.real(p[0, 0])

            evalues = np.linalg.eigvals(rho)
            n_0N_frac.append(max(evalues) / np.trace(rho))

        ts_frac = [ts[i][1] / ts[i][0] for i in range(ts.shape[0])]
        plt.scatter(ts_frac, n_0N_frac, s=3)
        plt.show()
        # plt.savefig('/condensate_fraction.png')


test = lattice(10)
#'manual' gives the a) spectrum, anything else gives the c) spectrum
# test.spectrum("manual", 0.5)
# test.spectrum("exact", 1)
test.condensate_frac()
