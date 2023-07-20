        # prepare plot
        G = Plotter(figsize=(6, 12), nrows=2, ncols=1)
        # colorlist
        colors = [mcolors.TABLEAU_COLORS[str(a)] for a in mcolors.TABLEAU_COLORS]

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
                #or np.real(x[1]) >= 1
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
            G.ax[i].set_ylim(0, 1.2)
            G.ax[i].set_title(
                r"Condensate fraction $\frac{n_0}{N}$ with "
                + str(matrix_type)
                + " matrices"
            )
            G.ax[i].set_xlabel(r"$s/t$")
            G.ax[i].set_ylabel(r"$\frac{n_0}{N}$")
            G.ax[i].legend()
        G.savefig(os.path.join(HOME_FOLDER, "..", "plots", f"condensate_fraction_{L}.png"))