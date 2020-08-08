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
import matplotlib.font_manager
import matplotlib as mpl


mpl.rc('font', family='Arial',size=9)

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
            plt.plot(x, y, label=label, lw=1, ls='--', color='black', alpha = 0.75)
        else:
            ax.plot(x, y, label=label, lw=1, ls='--', color='black', alpha = 0.75)

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
        fig_2 = plt.figure(constrained_layout=False, figsize=(6.29921, 9.52756))
    mpl.rc('font', family='Arial',size=9)
    fig_2.suptitle(f"BETSI Regression Diagnostics for {name}\n")

    # "Residual vs fitted" plot
    resid_vs_fit = fig_2.add_subplot(2, 2, 1)
    sns.residplot(fit_values, fit_resid, data=dataframe,
                  lowess=True,
                  scatter_kws={'alpha': .5, 'color': 'red'},
                  line_kws={'color': 'black', 'lw': 1, 'alpha': 0.75},
                  ax=resid_vs_fit)
    resid_vs_fit.axes.set
    resid_vs_fit.axes.set_title('Residuals vs Fitted',fontsize=11)
    resid_vs_fit.axes.set_xlabel('Fitted Values')
    resid_vs_fit.locator_params(axis='x', nbins=4)
    resid_vs_fit.axes.set_ylabel('Residuals')
    resid_vs_fit.tick_params(axis='both', which='major', labelsize=9)
    resid_vs_fit.tick_params(axis='both', which='minor', labelsize=9)

    dfit_values = (max(fit_values) - min(fit_values)) * 1
    resid_vs_fit.axes.set_xlim(min(fit_values) - dfit_values, max(fit_values) + dfit_values)
    dfit_resid = (max(fit_resid) - min(fit_resid)) * 1
    resid_vs_fit.axes.set_ylim(min(fit_resid) - dfit_resid, max(fit_resid) + dfit_resid)

    # "Normal Q-Q" plot
    QQ = ProbPlot(fit_stud_resid)

    qq_plot = QQ.qqplot(line='45',markerfacecolor='red',markeredgecolor='red',color='black', alpha=.3, lw=.5, ax=fig_2.add_subplot(2, 2, 2))
    qq_plot.axes[1].set_title('Normal Q-Q')
    qq_plot.axes[1].set_xlabel('Theoretical Quantiles')
    qq_plot.axes[1].set_ylabel('Studentized Residuals')
    qq_plot.axes[1].tick_params(axis='both', which='major')
    qq_plot.axes[1].tick_params(axis='both', which='minor')

    abs_norm_resid = np.flip(np.argsort(np.abs(fit_stud_resid)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        # Add annotations
        qq_plot.axes[0].annotate(i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], fit_stud_resid[i]), size = 9)

    # "Scale-location" plot
    scale_loc = fig_2.add_subplot(2, 2, 3)
    scale_loc.scatter(fit_values, fit_stud_resid_abs_sqrt, alpha=.5, c = 'red')
    sns.regplot(fit_values, fit_stud_resid_abs_sqrt, scatter=False, ci=False, lowess=True, line_kws={'color': 'black', 'lw': 1, 'alpha': .75}, ax=scale_loc)
    scale_loc.set_title('Scale-Location')
    scale_loc.set_xlabel('Fitted Values')
    scale_loc.set_ylabel('$\mathregular{\sqrt{|Studentized\ Residuals|}}$')
    scale_loc.tick_params(axis='both', which='major', labelsize=9)
    scale_loc.tick_params(axis='both', which='minor', labelsize=9)

    abs_sq_norm_resid = np.flip(np.argsort(fit_stud_resid_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        # Add annotations
        scale_loc.axes.annotate(i, xy=(fit_values[i], fit_stud_resid_abs_sqrt[i]), size=11)

    scale_loc.axes.set_xlim(min(fit_values) - .2 * max(fit_values), max(fit_values) + .2 * max(fit_values))
    scale_loc.locator_params(axis='x', nbins=4)

    # "Residuals vs leverage" plot
    res_vs_lev = fig_2.add_subplot(2, 2, 4)
    res_vs_lev.scatter(fit_leverage, fit_stud_resid, alpha=.5, color = 'red')
    sns.regplot(fit_leverage, fit_stud_resid, scatter=False, ci=False, lowess=True, line_kws={'color': 'black', 'lw': 1, 'alpha': .75}, ax=res_vs_lev)
    res_vs_lev.axes.set_title('Residuals vs Leverage')
    res_vs_lev.axes.set_xlabel('Leverage')
    res_vs_lev.axes.set_ylabel('Studentized Residuals')
    res_vs_lev.tick_params(axis='both', which='major')
    res_vs_lev.tick_params(axis='both', which='minor')

    leverage_top_3 = np.flip(np.argsort(fit_CD), 0)[:3]
    for i in leverage_top_3:
        # Add annotations
        res_vs_lev.axes.annotate(i, xy=(fit_leverage[i], fit_stud_resid[i]), size=9)

    p_3 = p[min_i:min_j]
    p_2 = len(fit.params)  # number of model parameters
    graph(lambda p_3: np.sqrt((.5 * p_2 * (1 - p_3)) / p_3), np.linspace(.001, max(fit_leverage), 50), 'Cook\'s Distance', ax=res_vs_lev)  # .5 line
    graph(lambda p_3: -1 * np.sqrt((.5 * p_2 * (1 - p_3)) / p_3), np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)
    graph(lambda p_3: np.sqrt((1 * p_2 * (1 - p_3)) / p_3), np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)  # 1 line
    graph(lambda p_3: -1 * np.sqrt((1 * p_2 * (1 - p_3)) / p_3), np.linspace(.001, max(fit_leverage), 50), ax=res_vs_lev)  # 1 line

    res_vs_lev.legend(prop={'size': 9})

    plt.subplots_adjust(bottom=0.07, top=0.91, hspace=.255, wspace=0.315, left=0.12, right=0.92)

    return fig_2


def create_matrix_plot(bet_filtered, rouq3, rouq4, name, fig=None):
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
        fig = plt.figure(figsize=(6.29921, 9.52756))

    fig.set_size_inches(6.29921, 9.52756)
    fig.suptitle(f"BETSI Analysis for {name}\n", fontname="Arial", fontsize = '11')
    fig.subplots_adjust(hspace=1.0, top=0.91, bottom=0.07, left=0.052, right=0.865, wspace=0.315)

    gs = gridspec.GridSpec(9, 2, figure=fig)

    # Plot "Adsorption isotherm"
    ax = fig.add_subplot(gs[:3,0])
    plot_isotherm(bet_filtered, ax)

    # Plot "Roquerol representation"
    ax = fig.add_subplot(gs[:3,1])
    plot_roquerol_representation(bet_filtered, ax)

    # Plot "Linear range"
    ax = fig.add_subplot(gs[3:6,0])
    plot_linear_y(bet_filtered, ax)

    # Plot "Filtered BET areas"
    if not rouq3 and not rouq4:
        ax = fig.add_subplot(gs[3:4, 1:])
        ax2 = fig.add_subplot(gs[4:6, 1:])
        # Plots a figure with a break in the y-axis
        plot_area_error_1(bet_filtered, ax, ax2)
    else:
        ax = fig.add_subplot(gs[3:6,1])
        # Plots a figure with NO break in the y-axis
        plot_area_error_2(bet_filtered, ax)

    # Plot "Filtered monolayer-loadings"
    ax = fig.add_subplot(gs[6:9,0])
    plot_monolayer_loadings(bet_filtered, ax)

    # Plot "Distribution of filtered BET areas"
    ax = fig.add_subplot(gs[6:9,1])
    plot_box_and_whisker(bet_filtered, ax)

    plt.tight_layout()
    return fig


def plot_isotherm(bet_filtered, ax=None):
    """ Plot the Isotherm alongside the selected linear region, spline interpolation, point corresponding to lowest error and fit from BET theory.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """
    if ax is None:
        # When this is not part of a larger plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    mpl.rc('font', family='Arial',size=9)
    ax.set_title(f"Adsorption Isotherm", fontname="Arial", fontsize = '11')
    ax.set_xlabel(r'$\mathregular{P/P_0}$')
    ax.set_ylabel(r'$\mathregular{N_2 uptake}$ (STP) $\mathregular{cm^3 g^{-1}}$')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, max(bet_filtered.q_adsorbed)])
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the isotherm itself
    ax.scatter(bet_filtered.pressure[:min_i], bet_filtered.q_adsorbed[:min_i], color='black', edgecolors='black', label='Adsorption Isotherm', alpha=0.50)
    ax.scatter(bet_filtered.pressure[min_j+1:], bet_filtered.q_adsorbed[min_j+1:], color='black', edgecolors='black',alpha = 0.50)

    # Plot the part corresponding to the selected linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1], bet_filtered.q_adsorbed[min_i:min_j + 1], marker='s',color='red', edgecolors='red', label='Linear Range', alpha=0.5)

    # plot pchip interpolation
    ax.plot(bet_filtered.x_range, pchip_interpolate(bet_filtered.pressure, bet_filtered.q_adsorbed, bet_filtered.x_range), color='black', alpha=.75, label='Pchip Interpolation')
    # plot corresponding pressure
    ax.scatter(bet_filtered.corresponding_pressure_pchip[min_i, min_j], bet_filtered.nm[min_i, min_j], marker='^', color='blue', edgecolor='blue', label='$\mathregular{N_m}$ Read', alpha=0.50)

    # Plot selected Monolayer loading (single point)
    ax.scatter(bet_filtered.calc_pressure[min_i, min_j], bet_filtered.nm[min_i, min_j], marker='v', color='green', edgecolors='green', edgecolor='green', label='$\mathregular{N_m}$ BET')

    # Plot the BET curve derived from BET theory
    ax.plot(bet_filtered.x_range, bet_filtered.bet_curve, c='g', label='BET Fit', alpha=.5)

    # Add a legend
    ax.autoscale(False)
    ax.legend(prop={'size': 9})


def plot_roquerol_representation(bet_filtered, ax=None):
    """ Plot the Roquerol representation with points corresponding to those in the selected linear region highlighted.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """

    if ax is None:
        # When this is not part of a larger plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    mpl.rc('font', family='Arial',size=9)
    ax.set_title(f"Rouquerol Representation", fontname="Arial", fontsize = '11')
    ax.set_xlabel(r'$\mathregular{P/P_0}$')
    ax.set_ylabel(r'$\mathregular{N(1-P/P_0)}$', fontname="Arial")
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the main Roquerol representation scatter
    ax.scatter(bet_filtered.pressure[:min_i], bet_filtered.rouq_y[:min_i], edgecolors='black', color='black', label=r'$\mathregular{N(1-P/P_0)}$', alpha=0.5)
    ax.scatter(bet_filtered.pressure[min_j+1:], bet_filtered.rouq_y[min_j+1:], edgecolors='black', color='black', alpha=0.5)

    # Plot the part corresponding to the linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1], bet_filtered.rouq_y[min_i:min_j + 1], marker='s', color='red', edgecolors='', label='Linear Range', alpha=0.50)

    # Add a legend
    ax.legend(prop={'size': 9})


def plot_linear_y(bet_filtered, ax=None):
    """ Plot the selected linear region of the linearised BET equation and print the formula of the
    straight line alongside it.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """
    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    mpl.rc('font', family='Arial',size=9)
    ax.set_title(f"Linear Range", fontname="Arial",fontsize='11')
    ax.set_xlabel(r'$\mathregular{P/P_0}$')
    ax.set_ylabel(r'$\mathregular{P/N(P_0 - P)}$', fontname="Arial", fontsize = '9')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the points in the selected linear region
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1], bet_filtered.linear_y[min_i:min_j + 1], marker='s', color='r', edgecolors='red', alpha = 0.50)

    # Plot the straight line obtained from the linear regression
    largest_valid_x = max(bet_filtered.pressure[min_i:min_j + 1])
    intercept_at_opt = bet_filtered.fit_intercept[min_i, min_j]
    grad_at_opt = bet_filtered.fit_grad[min_i, min_j]
    highest_valid_pressure = max(bet_filtered.pressure[min_i:min_j + 1])
    end_y = grad_at_opt * highest_valid_pressure + intercept_at_opt

    ax.plot([0, largest_valid_x], [intercept_at_opt, end_y], color='black',alpha=.75)

    # Set the plot limits
    smallest_y = 0.1 * min(bet_filtered.linear_y[min_i:min_j + 1])
    biggest_y = 1.1 * max(bet_filtered.linear_y[min_i:min_j + 1])
    smallest_x = 0.1 * min(bet_filtered.pressure[min_i:min_j + 1])
    biggest_x = 1.1 * max(bet_filtered.pressure[min_i:min_j + 1])

    ax.set_ylim(smallest_y, biggest_y)
    ax.set_xlim(smallest_x, biggest_x)

    # Print the equation of the straight line.
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
    y_eqn = r"y = {0:.8f}$x$ + {1:.8f}".format(bet_filtered.fit_grad[min_i, min_j], bet_filtered.fit_intercept[min_i, min_j])
    r_eqn = r"$R^2$ = {0:.8f}".format(bet_filtered.fit_rsquared[min_i, min_j])
    ax.text(0.05, 0.9, y_eqn, {'color': 'black', 'fontsize': 9}, transform=ax.transAxes)
    ax.text(0.05, 0.825, r_eqn, {'color': 'black', 'fontsize': 9}, transform=ax.transAxes)


def plot_area_error_1(bet_filtered, ax=None, ax2=None):
    """ Plot the distribution of valid BET areas, highlight those ending on the `knee` and print
    start and end indices.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    if ax is None:
        # When this is not part of a larger plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    mpl.rc('font', family='Arial',size=9)
    ax.set_title('Filtered BET areas ', fontname="Arial")
    ax2.set_xlabel(r'BET Area $\mathregular{m^2 g^{-1}}$', fontname="Arial", fontsize = '9')
    ax2.set_ylabel(r'Percentage Error %', fontname="Arial", fontsize = '9')
    ax2.yaxis.set_label_coords(-0.1, 0.78)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)
    ax2.tick_params(axis='both', which='major', labelsize=9)
    ax2.tick_params(axis='both', which='minor', labelsize=9)

    x_coords = bet_filtered.valid_bet_areas
    y_coords = bet_filtered.valid_pc_errors
    x_coords_nonvalid = np.array([x for x in x_coords if x not in bet_filtered.valid_knee_bet_areas])
    y_coords_nonvalid = np.array([y for y in y_coords if y not in bet_filtered.valid_knee_pc_errors])

    # Scatter plot of the Error across valid areas
    ax.scatter(x_coords_nonvalid, y_coords_nonvalid, color='red', edgecolors='red', picker=5, alpha =0.5)
    ax.scatter(bet_filtered.valid_knee_bet_areas, bet_filtered.valid_knee_pc_errors, color='b', edgecolors='b', marker='s', picker=5, alpha=0.5)
    ax.scatter(bet_filtered.bet_areas[min_i, min_j], bet_filtered.pc_error[min_i, min_j], marker='s', color='yellow', edgecolors='yellow')

    ax2.scatter(x_coords_nonvalid, y_coords_nonvalid, color='r', edgecolors='r', picker=5, alpha = 0.5)
    ax2.scatter(bet_filtered.valid_knee_bet_areas, bet_filtered.valid_knee_pc_errors, color='b', edgecolors='b', marker='s', picker=5, alpha=.5)
    ax2.scatter(bet_filtered.bet_areas[min_i, min_j], bet_filtered.pc_error[min_i, min_j], marker='s', color='orange', edgecolors='orange')

    # Axis settings
    ax.set_ylim(290, 310)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.tick_params(labeltop='off')
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

    # Add text annotation to each error point
    for i, type in enumerate(x_coords):
        index = f"({bet_filtered.valid_indices[0][i] + 1}," \
                f" {bet_filtered.valid_indices[1][i] + 1})"
        plt.text(x_coords[i], y_coords[i], index, fontsize=7, clip_on=True)


def plot_area_error_2(bet_filtered, ax=None):
    """ Plot the distribution of valid BET areas, highlight those ending on the `knee` and print start and end indices.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    if ax is None:
        # When this is not part of a larger plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    ax.set_title('Filtered BET areas ', fontname="Arial")
    ax.set_xlabel(r'BET Area $\mathregular{m^2 g^{-1}}$', fontname="Arial", fontsize = '9')
    ax.set_ylabel(r'Percentage Error %', fontname="Arial", fontsize = '9')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='major', labelsize=9)

    x_coords = bet_filtered.valid_bet_areas
    y_coords = bet_filtered.valid_pc_errors
    x_coords_nonvalid = np.array([x for x in x_coords if x not in bet_filtered.valid_knee_bet_areas])
    y_coords_nonvalid = np.array([y for y in y_coords if y not in bet_filtered.valid_knee_pc_errors])

    # Scatter plot of the Error across valid areas
    ax.scatter(x_coords_nonvalid, y_coords_nonvalid, color='red', edgecolors='red', picker=5, alpha=0.5)
    ax.scatter(bet_filtered.valid_knee_bet_areas, bet_filtered.valid_knee_pc_errors, color='b', edgecolors='b', marker='s', picker=5, alpha=0.50)
    ax.scatter(bet_filtered.bet_areas[min_i, min_j], bet_filtered.pc_error[min_i, min_j], marker='s', color='yellow', edgecolors='yellow')

    # Add text annotation to each error point.
    for i, type in enumerate(x_coords):
        index = f"({bet_filtered.valid_indices[0][i] + 1}," \
                f" {bet_filtered.valid_indices[1][i] + 1})"
        plt.text(x_coords[i], y_coords[i], index, fontsize=7, clip_on=True)

    # Set the Y-limit on the errors
    ax.set_ylim(0, max(y_coords) * 1.1)

def plot_monolayer_loadings(bet_filtered, ax=None):
    """ Plot the distribution of monolayer loadings alonside the Isotherm and fitted spline.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """

    if ax is None:
        # When this is not part of a larger plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details
    mpl.rc('font', family='Arial',size=9)
    ax.set_title("Filtered Monolayer-Loadings", fontname="Arial")
    ax.set_xlabel(r'$\mathregular{P/P_0}$', fontname="Arial", fontsize = '9')
    ax.set_ylabel(r'$\mathregular{N_2 uptake}$ (STP) $\mathregular{cm^3 g^{-1}}$')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    # Get min_i, min_j from the filtered result
    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    # Plot the Isotherm itself
    ax.scatter(bet_filtered.pressure[:min_i], bet_filtered.q_adsorbed[:min_i], color='black', edgecolors='black', alpha=0.5)
    ax.scatter(bet_filtered.pressure[min_j+1:], bet_filtered.q_adsorbed[min_j+1:], color='black', edgecolors='black', label='Adsorption Isotherm', alpha=0.5)
    # Plot the fitted spline
    ax.plot(bet_filtered.x_range, pchip_interpolate(bet_filtered.pressure, bet_filtered.q_adsorbed, bet_filtered.x_range), color='black',alpha=.5, label='Pchip Interpolation')

    # Plot the valid monolayer loadings.
    ax.scatter(bet_filtered.valid_calc_pressures, bet_filtered.valid_nm, marker='^', color='blue', edgecolors='blue', label='$\mathregular{N_m}$ valid', alpha=0.5)

    # Plot the valid optimum linear range
    ax.scatter(bet_filtered.pressure[min_i:min_j + 1], bet_filtered.q_adsorbed[min_i:min_j + 1], marker='s', color='red', edgecolors='red', label='Linear Range', alpha=0.5)

    # Plot the single optimum Monolayer loading
    ax.scatter(bet_filtered.calc_pressure[min_i, min_j], bet_filtered.nm[min_i, min_j], marker='s', color='orange', edgecolors='orange', label='N$_m$ BET')

    # Plot the Fit obtained from the BET equation
    ax.plot(bet_filtered.x_range, bet_filtered.bet_curve, c='g',alpha=.5, label='BET Fit')

    # Set the Xlimits and add a legend
    ax.set_xlim([-0.001, max(bet_filtered.valid_calc_pressures)])
    ax.set_ylim([0.0, max(bet_filtered.q_adsorbed)])

    # Add a legend
    ax.autoscale(False)
    ax.legend(loc=4, prop={'size': 9})


def plot_box_and_whisker(bet_filtered, ax=None):
    """ Plot a box and whisker plot for the valid BET areas.

    Args:
        bet_filtered: A BETFilterAppliedResults object.
        ax: Optional matplotlib axis object. If none is provided, an axis for a single subplot is made.

    """
    if ax is None:
        # When this is not part of a larger plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # Set axis details.
    mpl.rc('font', family='Arial',size=9)
    ax.set_title('Distribution of filtered BET Areas', fontname="Arial")
    ax.set_ylabel(r'BET Area $\mathregular{m^2 g^{-1}}$', fontname="Arial", fontsize = '9')
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.tick_params(axis='both', which='minor', labelsize=9)

    min_i = bet_filtered.min_i
    min_j = bet_filtered.min_j

    y_min = [bet_filtered.bet_areas[min_i, min_j]]
    x_min = np.random.normal(1, 0.04, 1)

    y = list(set(bet_filtered.valid_knee_bet_areas) - set(y_min))
    x = np.random.normal(1, 0.04, size=len(y))

    y_2 = list(set(bet_filtered.valid_bet_areas) - set(y) - set(y_min))
    x_2 = np.random.normal(1, 0.04, size=len(y_2))

    # Plot the filtered BET areas
    ax.scatter(x_2, y_2, alpha=0.5, color='red', edgecolor='red')
    ax.scatter(x, y, color='blue', edgecolor='blue', alpha=0.5)
    ax.scatter(x_min, y_min, marker='s', color='orange', edgecolor='orange')

    if len(x)==0:
        ax.set_xlim([.75,1.25])
        dy = y_min[0]*0.25
        ax.set_ylim(y_min[0] - dy, y_min[0] + dy)
    else:
        ax.set_xlim([.75, 1.25])

        if len(y_2)==0:
                dy = (max(y)-min(y)) * 1
                ax.set_ylim(min(y) - dy, max(y) + dy)
        else:
            dy = (max(y_2) - min(y_2)) * 1
            ax.set_ylim(min(y_2) - dy, max(y_2) + dy)

    # Make the boxplot of valid areas
    medianprops = dict(linestyle='--', linewidth=1, color='black', alpha=0.35) # median line properties
    ax.boxplot(bet_filtered.valid_bet_areas, showfliers=False, medianprops=medianprops)
    ax.set_xticks([])

    # Write BET area
    called_BET_area = """BET Area = {0:0.0f} $m^2/g$""".format(np.around((bet_filtered.bet_areas[min_i, min_j])), decimals=0, out=None)
    ax.text(0.05, 0.90, called_BET_area, {'color': 'black', 'fontsize': 9}, transform=ax.transAxes)
