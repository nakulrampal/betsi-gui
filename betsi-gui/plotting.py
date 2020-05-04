"""
Defines a series of plotting functions that can be produced directly from a filtered BET results
 object.

The plots can all be made individually or as part of the larger 2x3 plot matrix.
"""
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import splev, pchip_interpolate
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.gridspec as gridspec


def regression_diagnostics_plots(bet_filtered, name, fig_2=None):
    """ Creates 4 regression diagnostics plots in 2 x 2 matrix
    Args:
        fit: Matplotlib Figure
        bet_filtered : A BETFilterAppliedResults object
        name: A string, name to give as a title.
        
    Returns:
        Fig, the updated matplotlib figure
        
    """
    # Obtaining Regression Diagnostics

    # Gather data and put in DF
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j + 1
    p = bet_filtered.pressure
    lin_q = bet_filtered.linear_y
    P = pd.DataFrame(p)
    LIN_Q = pd.DataFrame(lin_q)
    dataframe = pd.concat([P, LIN_Q], axis=1)

    # Helper functions

    num_points = len(p)

    def graph(formula, x_range, label=None, ax=None):
        """Helper function for plotting cook Distance lines
        """
        x = x_range
        y = formula(x)
        if ax is None:
            plt.plot(x, y, label=label, lw=1, ls='--', color='red')
        else:
            ax.plot(x, y, label=label, lw=1, ls='--', color='red')

    # OLS regression
    x = sm.add_constant(p)
    model = sm.OLS(lin_q[min_i:min_j], x[min_i:min_j])
    fit = model.fit()
    fit_values = fit.fittedvalues
    fit_resid = fit.resid
    fit_stud_resid = fit.get_influence().resid_studentized_internal
    fit_stud_resid_abs_sqrt = np.sqrt(np.abs(fit_stud_resid))
    fit_abs_resid = np.abs(fit_resid)
    fit_leverage = fit.get_influence().hat_matrix_diag
    fit_CD = fit.get_influence().cooks_distance[0]

    # Make new figure
    if fig_2 is None:
        fig_2 = plt.figure(figsize=(10, 9))

    # fig_2.suptitle(f"{name} Regression Diagnostics")

    # Residual vs fitted
    resid_vs_fit = fig_2.add_subplot(2, 2, 1)
    sns.residplot(fit_values, fit_resid, data=dataframe,
                  lowess=True,
                  scatter_kws={'alpha': .5},
                  line_kws={'color': 'red', 'lw': 1, 'alpha': .8},
                  ax=resid_vs_fit)
    resid_vs_fit.axes.set
    resid_vs_fit.axes.set_title('Residuals vs Fitted')
    resid_vs_fit.axes.set_xlabel('Fitted Values')
    resid_vs_fit.locator_params(axis='x', nbins=4)
    resid_vs_fit.axes.set_ylabel('Residuals')

    dfit_values = (max(fit_values) - min(fit_values)) * 1
    resid_vs_fit.axes.set_xlim(min(fit_values) - dfit_values, max(fit_values) + dfit_values)
    dfit_resid = (max(fit_resid) - min(fit_resid)) * 1
    resid_vs_fit.axes.set_ylim(min(fit_resid) - dfit_resid, max(fit_resid) + dfit_resid)

    # QQplot
    QQ = ProbPlot(fit_stud_resid)

    qq_plot = QQ.qqplot(line='45', alpha=.5, lw=1, ax=fig_2.add_subplot(2, 2, 2))
    qq_plot.axes[1].set_title('Normal Q-Q')
    qq_plot.axes[1].set_xlabel('Theoretical Quantiles')
    qq_plot.axes[1].set_ylabel('Studentized Residuals')

    abs_norm_resid = np.flip(np.argsort(np.abs(fit_stud_resid)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        qq_plot.axes[0].annotate(i,
                                 xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                                     fit_stud_resid[i]))
        # Scale Location plot
    scale_loc = fig_2.add_subplot(2, 2, 3)
    scale_loc.scatter(fit_values, fit_stud_resid_abs_sqrt, alpha=.5)
    sns.regplot(fit_values, fit_stud_resid_abs_sqrt,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': .8}, ax=scale_loc)
    scale_loc.set_title('Scale-Location')
    scale_loc.set_xlabel('Fitted Values')
    scale_loc.set_ylabel('$\sqrt{|Studentized\ Residuals|}$')

    abs_sq_norm_resid = np.flip(np.argsort(fit_stud_resid_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        scale_loc.axes.annotate(i,
                                xy=(fit_values[i],
                                    fit_stud_resid_abs_sqrt[i]))

    scale_loc.axes.set_xlim(min(fit_values) - .2 * max(fit_values), max(fit_values) + .2 * max(fit_values))
    scale_loc.locator_params(axis='x', nbins=4)
    # plt.tight_layout()

    # Residuals vs leverage
    res_vs_lev = fig_2.add_subplot(2, 2, 4)
    res_vs_lev.scatter(fit_leverage, fit_stud_resid, alpha=.5)
    sns.regplot(fit_leverage, fit_stud_resid,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': .8},
                ax=res_vs_lev)
    # res_vs_lev.axes[0].set_xlim(0, max(fit_leverage)*.01)
    # res_vs_lev.axes[0].set_ylim(-3,5)
    res_vs_lev.axes.set_title('Residuals vs Leverage')
    res_vs_lev.axes.set_xlabel('Leverage')
    res_vs_lev.axes.set_ylabel('Studentized Residuals')

    leverage_top_3 = np.flip(np.argsort(fit_CD), 0)[:3]
    for i in leverage_top_3:
        res_vs_lev.axes.annotate(i,
                                 xy=(fit_leverage[i],
                                     fit_stud_resid[i]))

    p_3 = p[min_i:min_j]
    p_2 = len(fit.params)  # number of model parameters
    graph(lambda p_3: np.sqrt((.5 * p_2 * (1 - p_3)) / p_3),
          np.linspace(.001, max(fit_leverage), 50),
          'Cook\'s Distance', ax=res_vs_lev)  # .5 line

    graph(lambda p_3: -1 * np.sqrt((.5 * p_2 * (1 - p_3)) / p_3),
          np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)

    graph(lambda p_3: np.sqrt((1 * p_2 * (1 - p_3)) / p_3),
          np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)  # 1 line
    graph(lambda p_3: -1 * np.sqrt((1 * p_2 * (1 - p_3)) / p_3),
          np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)  # 1 line
    res_vs_lev.legend()

    plt.subplots_adjust(bottom=0.05, top=0.95, hspace=.2)

    # plt.tight_layout()
    return fig_2


def create_matrix_plot(bet_filtered, name, fig=None):
    """ Creates all 6 of the key plots in a 2x3 matrix

    Args:
        fig: Matplotlib Figure
        bet_filtered: A BETFilterAppliedResults object.
        name: A string, name to give as a title.

    Returns:
        Fig, the updated matplotlib figure

    """
    # Make Isotherm Plot
    if fig is None:
        fig = plt.figure(constrained_layout=True, figsize=(9,9))

    fig.suptitle(f"{name} Analysis\n")
    #fig.subplots_adjust(hspace=0.90)

    gs = gridspec.GridSpec(6, 3, figure=fig)

    ax = fig.add_subplot(gs[:3,0])
    plot_isotherm(bet_filtered, ax)

    ax = fig.add_subplot(gs[:3,1])
    plot_roquerol_representation(bet_filtered, ax)

    ax = fig.add_subplot(gs[:3,2])
    plot_linear_y(bet_filtered, ax)

    ax = fig.add_subplot(gs[3,0])
    ax2 = fig.add_subplot(gs[4:,0])
    plot_area_error(bet_filtered, ax, ax2)

    ax = fig.add_subplot(gs[3:,1])
    plot_monolayer_loadings(bet_filtered, ax)

    ax = fig.add_subplot(gs[3:,2])
    plot_box_and_whisker(bet_filtered, ax)

    return fig


def plot_isotherm(bet_filtered, ax=None):
    """ Plot the Isotherm alongside the selected linear region, spline interpolation, point
    corresponding to lowest error and fit from BET theory.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """

    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title(f"Adsorption Isotherm")
    ax.set_xlabel(r'$P/P_0$')
    ax.set_ylabel(r'$N_2$ uptake (STP) $cm^3$ $g^{-1}$')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, max(bet_filtered.q_adsorbed)])

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the isotherm itself.
    ax.scatter(bet_filtered.pressure,
               bet_filtered.q_adsorbed, marker='D', edgecolors='black', label='Adsorption Isotherm')

    # Plot the part corresponding to the selected linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1],
               bet_filtered.q_adsorbed[min_i:min_j + 1],
               marker='D', color='red', edgecolors='black', label='Linear Range')

    # Plot the spline interpolation
    # ax.plot(bet_filtered.x_range, splev(bet_filtered.x_range,
    # bet_filtered.fitted_spline,
    # der=0, ext=0), color='magenta', label='Spline-interpolation')
    # plot pchip interpolation
    ax.plot(bet_filtered.x_range,
            pchip_interpolate(bet_filtered.pressure, bet_filtered.q_adsorbed, bet_filtered.x_range), color='magenta',
            label='Pchip Interpolation')
    # plot corresponding pressure
    ax.scatter(bet_filtered.corresponding_pressure_pchip[min_i, min_j], bet_filtered.nm[min_i, min_j], marker='D',
               color='black', label='Vm Read')

    # Plot selected Monolayer loading (single point)
    ax.scatter(bet_filtered.calc_pressure[min_i, min_j],
               bet_filtered.nm[min_i, min_j],
               marker='D', color='yellow', edgecolors='black', label='Vm BET')

    # Plot the BET curve derived from BET theory
    ax.plot(bet_filtered.x_range, bet_filtered.bet_curve, c='g', label='BET Fit')

    # Add a legend.
    ax.autoscale(False)
    ax.legend(prop={'size': 8})


def plot_roquerol_representation(bet_filtered, ax=None):
    """ Plot the Roquerol representation with points corresponding to those in the selected
    linear region highlighted.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """

    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title(f"Rouquerol Representation")
    ax.set_xlabel(r'$P/P_0$')
    ax.set_ylabel(r'$V(P_0 - P)$')

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the main Roquerol representation scatter
    ax.scatter(bet_filtered.pressure,
               bet_filtered.rouq_y, marker='D', edgecolors='black', label=r'$V(P_0 - P)$')

    # Plot the part corresponding to the linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1],
               bet_filtered.rouq_y[min_i:min_j + 1],
               marker='D', color='red', edgecolors='black', label='Linear Range')

    ax.legend(prop={'size': 8})


def plot_linear_y(bet_filtered, ax=None):
    """ Plot the selected linear region of the linearised BET equation and print the formula of the
    straigh line alongside it.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """
    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title(f"Linear Range")
    ax.set_xlabel(r'$P/P_0$')
    ax.set_ylabel(r'$P/V(P_0 - P)$')

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the points in the selected linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1],
               bet_filtered.linear_y[min_i:min_j + 1],
               marker='D', color='r', edgecolors='black')

    # Plot the straight line obtained from the linear regression
    largest_valid_x = max(bet_filtered.pressure[min_i:min_j + 1])
    intercept_at_opt = bet_filtered.fit_intercept[min_i, min_j]
    grad_at_opt = bet_filtered.fit_grad[min_i, min_j]
    highest_valid_pressure = max(bet_filtered.pressure[min_i:min_j + 1])
    end_y = grad_at_opt * highest_valid_pressure + intercept_at_opt

    ax.plot([0, largest_valid_x],
            [intercept_at_opt, end_y], color='black')

    # Set the plot limits
    smallest_y = 0.1 * min(bet_filtered.linear_y[min_i:min_j + 1])
    biggest_y = 1.1 * max(bet_filtered.linear_y[min_i:min_j + 1])
    smallest_x = 0.1 * min(bet_filtered.pressure[min_i:min_j + 1])
    biggest_x = 1.1 * max(bet_filtered.pressure[min_i:min_j + 1])

    ax.set_ylim(smallest_y, biggest_y)
    ax.set_xlim(smallest_x, biggest_x)

    # Print the equation of the straight line.
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
    y_eqn = r"y = {0:.8f}$x$ + {1:.8f}".format(bet_filtered.fit_grad[min_i, min_j],
                                               bet_filtered.fit_intercept[min_i, min_j])
    r_eqn = r"$R^2$ = {0:.8f}".format(bet_filtered.fit_rsquared[min_i, min_j])
    ax.text(0.05, 0.9, y_eqn, {'color': 'black',
                               'fontsize': 8}, transform=ax.transAxes)
    ax.text(0.05, 0.85, r_eqn, {'color': 'black',
                                'fontsize': 8}, transform=ax.transAxes)


def plot_area_error(bet_filtered, ax=None, ax2=None):
    """ Plot the distribution of valid BET areas, highlight those ending on the `knee` and print
    start and end indices.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title('Filtered BET areas ')
    ax2.set_xlabel(r'BET Area $m^2 g^{-1}$')
    ax2.set_ylabel(r'Percentage Error %')

    x_coords = bet_filtered.valid_bet_areas
    y_coords = bet_filtered.valid_pc_errors
    x_coords_nonvalid = np.array(
        [x for x in x_coords if x not in bet_filtered.valid_knee_bet_areas])
    y_coords_nonvalid = np.array(
        [y for y in y_coords if y not in bet_filtered.valid_knee_pc_errors])

    # Scatter plot of the Error across valid areas
    ax.scatter(x_coords_nonvalid, y_coords_nonvalid, color='r',
               edgecolors='black', marker='D', picker=5)
    ax.scatter(bet_filtered.valid_knee_bet_areas,
               bet_filtered.valid_knee_pc_errors, color='b', edgecolors='black', marker='D', picker=5)
    ax.scatter(bet_filtered.bet_areas[min_i, min_j], bet_filtered.pc_error[min_i, min_j], marker='D', color='yellow',
               edgecolors='black')

    ax2.scatter(x_coords_nonvalid, y_coords_nonvalid, color='r',
               edgecolors='black', marker='D', picker=5)
    ax2.scatter(bet_filtered.valid_knee_bet_areas,
               bet_filtered.valid_knee_pc_errors, color='b', edgecolors='black', marker='D', picker=5)
    ax2.scatter(bet_filtered.bet_areas[min_i, min_j], bet_filtered.pc_error[min_i, min_j], marker='D', color='yellow',
               edgecolors='black')

    ax.set_ylim(300, 300)
    ax2.set_ylim(0, 40)

    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(labeltop='off')# don't put tick labels at the top
    ax2.tick_params(labeltop='off')
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Add text annotation to each error point.
    for i, type in enumerate(x_coords):
        index = f"({bet_filtered.valid_indices[0][i] + 1}," \
                f" {bet_filtered.valid_indices[1][i] + 1})"
        plt.text(x_coords[i], y_coords[i], index, fontsize=7, clip_on=True)

    # Set the Y-limit on the errors
    #ax.set_ylim(0, max(y_coords) * 1.1)


def plot_monolayer_loadings(bet_filtered, ax=None):
    """ Plot the distribution of monolayer loadings alonside the Isotherm and fitted spline.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """

    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title("Filtered Monolayer-Loadings")
    ax.set_xlabel(r'$P/P_0$')
    ax.set_ylabel(r'$N_2$ uptake (STP) $cm^3$ $g^{-1}$')

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the Isotherm itself
    ax.scatter(bet_filtered.pressure,
               bet_filtered.q_adsorbed,
               marker='D', color='blue', edgecolors='black', label='Adsorption Isotherm')

    # Plot the fitted spline.
    ax.plot(bet_filtered.x_range,
            pchip_interpolate(bet_filtered.pressure, bet_filtered.q_adsorbed, bet_filtered.x_range), color='magenta',
            label='Pchip Interpolation')

    # Plot the valid monolayer loadings.
    ax.scatter(bet_filtered.valid_calc_pressures,
               bet_filtered.valid_nm,
               marker='D', facecolors='lightgreen', edgecolors='black',
               label='Valid monolayer loading')

    # Plot the valid optimum linear range
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1],
               bet_filtered.q_adsorbed[min_i:min_j + 1],
               marker='D', color='red', edgecolors='black', label='Linear Range')

    # Plot the single optimum Monolayer loading
    ax.scatter(bet_filtered.calc_pressure[min_i, min_j],
               bet_filtered.nm[min_i, min_j],
               marker='D', color='yellow', edgecolors='black', label='Vm BET')

    # Plot the Fit obtained from the BET equation
    ax.plot(bet_filtered.x_range, bet_filtered.bet_curve, c='g', label='BET Fit')

    # Set the Xlimits and add a legend
    ax.set_xlim([-0.001, max(bet_filtered.valid_calc_pressures) * 1.2])
    ax.set_ylim([0.0, max(bet_filtered.q_adsorbed) * 1.1])

    ax.autoscale(False)
    ax.legend(prop={'size': 8})


def plot_box_and_whisker(bet_filtered, ax=None):
    """ Plot a box and whisker plot for the valid BET areas.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is
        made.

    """
    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    ax.set_title('Distribution of filtered BET Areas')
    ax.set_ylabel(r'BET Area $m^2 g^{-1}$')

    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    y_min = [bet_filtered.bet_areas[min_i, min_j]]
    x_min = np.random.normal(1, 0.04, 1)

    y = list(set(bet_filtered.valid_knee_bet_areas) - set(y_min))
    x = np.random.normal(1, 0.04, size=len(y))
    ax.scatter(x, y, marker='D', color='blue', edgecolor='black', alpha=0.5)

    y = list(set(bet_filtered.valid_bet_areas) - set(y) - set(y_min))
    x = np.random.normal(1, 0.04, size=len(y))
    ax.scatter(x, y, alpha=0.5, marker='D', color='red', edgecolor='black')

    ax.scatter(x_min, y_min, marker='D', color='yellow', edgecolor='black')

    dx = (max(x) - min(x)) * 1
    ax.set_xlim(min(x) - dx, max(x) + dx)
    dy = (max(y) - min(y)) * 1
    ax.set_ylim(min(y) - dy, max(y) + dy)

    # Make the boxplot of valid areas
    ax.boxplot(bet_filtered.valid_bet_areas, showfliers=False)

    # Write BET area
    called_BET_area = """BET Area =
{0:0.0f} $m^2/g$""".format(np.around((bet_filtered.bet_areas[min_i, min_j])), decimals=0, out=None)
    ax.text(0.05, 0.85, called_BET_area, {'color': 'black',
                                          'fontsize': 8}, transform=ax.transAxes)
