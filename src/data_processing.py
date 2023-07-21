import numpy as np
import matplotlib.colors as mcolors

from plotter import Plotter

import sys, os
import glob

HOME_FOLDER = os.path.abspath(os.path.dirname(__file__))

colors = [mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS]
colors.extend(['k','m','b'])

def plot_manual() -> None:

    evals_new = []
    s_t = []

    dir = os.path.join(HOME_FOLDER,"..", 'data')

    L = 2

    for filename in glob.glob(os.path.join(dir, "s_t",'*_manual.csv')):
        s_t.append(np.loadtxt(filename, delimiter = ','))

    for filename in glob.glob(os.path.join(dir, "evals_new",'*_manual.csv')):
        evals_new.append(np.loadtxt(filename, delimiter = ','))

    P = Plotter(figsize=(6*2, 4*5), nrows=5, ncols=2, usetex=True)
    
    for i in range(5):
        for j in range(2):
            L = i*2+j+1

            P.ax[i][j].plot( 
                s_t[L], evals_new[L][0],
                label=f"EV {0}", color="k", lw=1.5,
            )
            P.ax[i][j].plot(
                s_t[L], evals_new[L][1],
                label=f"all other EV's", color="b", lw=0.7,
            )
            for k in range(2, evals_new[L].shape[0] - 1):
                # plot without label for the repeats
                P.ax[i][j].plot(
                    s_t[L], evals_new[L][k], color="b", lw=0.7,
                )

            # last EV is magenta
            P.ax[i][j].plot(
                s_t[L],
                evals_new[L][-1],
                label="Last EV",
                color="m",
                lw=1.5,
            )

            P.ax[i][j].legend()
            P.ax[i][j].set_xscale("log")
            P.ax[i][j].set_title(
                "Spectrum manual, "
                + "sparse matrices"
                + f", System size: {L+1}"
            )
            if i == 4:
                P.ax[i][j].set_xlabel(r"$s/t$")
            if j == 0:
                P.ax[i][j].set_ylabel(r"$E_n$")

        #P.fig.subplots_adjust(hspace = 0.5, vspace = 0.5)
        P.fig.tight_layout(pad = 0.5)

        P.savefig(os.path.join(HOME_FOLDER, "..", "plots", "spectrum_manual.pdf"))

def plot_exact(manual_subset: bool) -> None:
    evals_new = []
    s_t = []

    dir = os.path.join(HOME_FOLDER,"..", 'data')

    for filename in glob.glob(os.path.join(dir, "s_t",'*_exact.csv')):
        s_t.append(np.loadtxt(filename, delimiter = ','))

    for filename in glob.glob(os.path.join(dir, "evals_new",'*_exact.csv')):
        evals_new.append(np.loadtxt(filename, delimiter = ','))

    # option to only plot the first two plots, since they need different ylims to be sensible
    if manual_subset == False:
        rows = 5
        savepath = os.path.join(HOME_FOLDER, "..", "plots", "spectrum_exact.pdf")
    else:
        rows = 1
        savepath = os.path.join(HOME_FOLDER, "..", "plots", "spectrum_exact_manual_subset.pdf")

    P = Plotter(figsize=(6*2, 4*rows), nrows=rows, ncols=2, usetex=True, squeeze = False)
    
    for i in range(rows):
        for j in range(2):
            # this L is not the system size, but the list index. the list was started at L = 2, therefore the "true" L is L+2 (0 indexing as well)
            L = i*2+j

            P.ax[i][j].plot( 
                s_t[L], evals_new[L][0],
                label=f"EV {0}", color="k", lw=1.5,
            )
            P.ax[i][j].plot(
                s_t[L], evals_new[L][1],
                label=f"all other EV's", color="b", lw=0.7,
            )
            for k in range(2, evals_new[L].shape[0] - 1):
                # plot without label for the repeats

                #smaller linewidth for the highyl crowded plots
                if L >=7:
                    P.ax[i][j].plot(
                        s_t[L], evals_new[L][k], color="b", lw=0.1,
                    )
                else:
                    P.ax[i][j].plot(
                        s_t[L], evals_new[L][k], color="b", lw=0.7,
                    )

            # last EV is magenta
            P.ax[i][j].plot(
                s_t[L],
                evals_new[L][-1],
                label="Last EV",
                color="m",
                lw=1.5,
            )

            P.ax[i][j].legend()
            P.ax[i][j].set_ylim(-60//5*rows,60//5*rows)
            P.ax[i][j].set_xscale("log")
            P.ax[i][j].set_title(
                "Spectrum exact, "
                + "sparse matrices"
                + f", System size: {L+2}"
            )
            if i == 4:
                P.ax[i][j].set_xlabel(r"$s/t$")
            if j == 0:
                P.ax[i][j].set_ylabel(r"$E_n$")

    P.fig.tight_layout(pad = 0.5)
    P.savefig(savepath)
        

def plot_cf() -> None:

    G = Plotter(figsize=(8, 10), nrows=2, ncols=1, sharex = True)

    values = np.array([])

    dir = os.path.join(HOME_FOLDER,"..", 'data', 'CF')

    rho_values = np.array([])

    L = 2

    # first gather all unique rho values to color after
    for filename in glob.glob(os.path.join(dir,'*.csv')):
        values = np.loadtxt(filename, delimiter = ',')

        for x in values:
            rho_values = np.append(rho_values,x[2])
        rho_values = np.unique(rho_values)

    for filename in glob.glob(os.path.join(dir,'*.csv')):
        values = np.loadtxt(filename, delimiter = ',')

        # keep track of the rhos already plotted with legend
        L_already_labeled = []
        # keep track of the rhos already plotted with legend
        rho_already_labeled = []
        # plot condensation frac against s_t and color according to rho
        for x in values:
            # check for nan's and infs (low L's may have 0 particles in them)
            if (
                np.any(np.isnan(x[1])) == True
                or np.any(np.isinf(x[1])) == True
                or int(x[2]) >= 1
            ):
                continue
            else:
                # odd L's
                if np.real(L) % 2 != 0:
                    # label with rho and L values and choose new color if rho is new
                    if x[2] not in rho_already_labeled:
                        x = np.real(x)
                        G.ax[0].scatter(
                            x[0],
                            x[1],
                            s=4,
                            label=r"$\rho$ = %.3f" % (x[2]) + f", L = {int(L)}",
                            color=colors[np.where(rho_values == x[2])[0][0]],
                        )
                        rho_already_labeled.append(x[2])
                    # plot unlabeled if rho already appeared
                    else:
                        G.ax[0].scatter(
                            x[0],
                            x[1],
                            s=4,
                            color=colors[np.where(rho_values == x[2])[0][0]],
                        )
                # even L's
                else:
                    # label with rho and L values and choose new color if rho is new
                    if L not in L_already_labeled:
                        x = np.real(x)
                        G.ax[1].scatter(
                            x[0],
                            x[1],
                            s=4,
                            label=r"$\rho$ = %.3f" % (x[2]) + f", L = {int(L)}",
                            color=colors[int(L)],
                        )
                        L_already_labeled.append(L)
                    # plot unlabeled if rho already appeared
                    else:
                        G.ax[1].scatter(
                            x[0],
                            x[1],
                            s=4,
                            color=colors[int(L)],
                        )
        L += 1 

    # Plotting    

    G.fig.tight_layout(pad = 0.5)
    G.ax[1].set_xlabel(r"$s/t$")
    G.ax[0].set_title(
            r"Condensate fraction $\frac{n_0}{N}$ with "
            + "dense"
            + " matrices"
        )
    
    for i in range(2):
        G.ax[i].set_xscale("log")
        G.ax[i].set_ylim(0.4, 1.2)
        
        G.ax[i].set_ylabel(r"$\frac{n_0}{N}$")

        # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-outside-the-plot
        box = G.ax[i].get_position()
        G.ax[i].set_position([box.x0, box.y0, box.width * 0.8,box.height])

        G.ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    G.savefig(os.path.join(HOME_FOLDER, "..", "plots", f"condensate_fraction.pdf"),bbox_inches="tight")

if __name__ == "__main__":
    plot_manual()
    plot_exact(manual_subset = True)
    plot_exact(manual_subset = False)
    plot_cf()