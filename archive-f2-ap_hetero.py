import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from seaborn import pointplot, swarmplot


plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_figure(is_chamber_merged=False):
    fig = plt.figure(figsize=(6.5, 6))
    fig.subplots_adjust(.1, .05, .95, .95)

    grid = fig.add_gridspec(3, 4, hspace=.2, wspace=0.3)
    subgrid = grid[2,0:3].subgridspec(2, 1, height_ratios=[1, 2])
    ax_large = fig.add_subplot(subgrid[0])
    ax_small = fig.add_subplot(subgrid[1])
    
    axs = [fig.add_subplot(grid[0, 0:3]), 
            fig.add_subplot(grid[1, 0:3]),
            ax_small,
            ax_large
            ]

    ax_label = fig.add_subplot(grid[:, 3])

    plot_ap_feature_vs_lit(axs[0], 'MP', is_chamber_merged)
    plot_ap_feature_vs_lit(axs[1], 'APD90', is_chamber_merged)
    #plot_ap_feature_vs_lit(axs[2], 'APA')
    #plot_ap_feature_vs_lit(axs[2], 'dVdt')
    plot_dVdt(ax_small, ax_large, is_chamber_merged)
    ax_small.set_ylim(-5, 75)
    ax_large.set_ylim(100, 300)

    plot_literature_names(ax_label, is_chamber_merged)

    matplotlib.rcParams['pdf.fonttype'] = 42

    letters = ['A', 'B', '', 'C']

    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(letters[i], y=.94, x=-.1)

    ax_large.spines['bottom'].set_visible(False)
    ax_large.set_xticks([], [])

    kwargs = dict(transform=ax_large.transAxes, color='k', clip_on=False)
    d = .015
    ax_large.plot((-d, +d), (-d*2, +d*2), **kwargs)        # top-left diagonal
    #ax_large.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_small.transAxes)  # switch to the bottom axes
    d = .015
    ax_small.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax_small.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.savefig('./figure-pdfs/f-lit_ap_heterogeneity.pdf')#, transparent=True)
    plt.show()


def plot_two_panel(is_chamber_merged=False):
    fig = plt.figure(figsize=(6.5, 5))
    fig.subplots_adjust(.1, .05, .95, .95)

    grid = fig.add_gridspec(2, 4, hspace=.2, wspace=0.3)
    subgrid = grid[1,0:3].subgridspec(2, 1, height_ratios=[1, 2])
    ax_large = fig.add_subplot(subgrid[0])
    ax_small = fig.add_subplot(subgrid[1])
    
    axs = [fig.add_subplot(grid[0, 0:3]), 
            #fig.add_subplot(grid[1, 0:3]),
            ax_small,
            ax_large
            ]

    ax_label = fig.add_subplot(grid[:, 3])

    plot_ap_feature_vs_lit(axs[0], 'MP', is_chamber_merged)
    #plot_ap_feature_vs_lit(axs[1], 'APD90', is_chamber_merged)
    #plot_ap_feature_vs_lit(axs[2], 'APA')
    #plot_ap_feature_vs_lit(axs[2], 'dVdt')
    plot_dVdt(ax_small, ax_large, is_chamber_merged)
    ax_small.set_ylim(-5, 75)
    ax_large.set_ylim(100, 300)

    plot_literature_names(ax_label, is_chamber_merged)

    matplotlib.rcParams['pdf.fonttype'] = 42

    letters = ['A', 'B', '', 'C']

    for i, ax in enumerate(axs):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(letters[i], y=.94, x=-.1)

    ax_large.spines['bottom'].set_visible(False)
    ax_large.set_xticks([], [])

    kwargs = dict(transform=ax_large.transAxes, color='k', clip_on=False)
    d = .015
    ax_large.plot((-d, +d), (-d*2, +d*2), **kwargs)        # top-left diagonal
    #ax_large.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=ax_small.transAxes)  # switch to the bottom axes
    d = .015
    ax_small.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    #ax_small.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    plt.savefig('./figure-pdfs/f-lit_ap_heterogeneity2p.pdf')#, transparent=True)
    plt.show()


def plot_ap_feature_vs_lit(ax, feature_name, is_chamber_merged):
    lit_dat = pd.read_excel('./data/literature_ap_morph.xlsx', sheet_name='Sheet2', header=0)

    if is_chamber_merged:
        lit_dat = merge_chamber_dat(lit_dat)
    num_exps = lit_dat.shape[0] + 2
    exp_dat = pd.read_csv('./data/ap_features.csv')
    print(exp_dat.columns)

    lit_dat.sort_values('MP', inplace=True, ascending=False)

    x = np.linspace(1, lit_dat[feature_name].shape[0]+1, lit_dat[feature_name].shape[0]+1)

    i = 0
    exp_mean_mp = exp_dat['MP'].mean()
    for val in lit_dat['MP'].values:
        if val < exp_mean_mp:
            break
        i += 1

    y_vals = [v for v in lit_dat[feature_name].values]
    y_vals = y_vals[0:i] + [exp_dat[feature_name].mean()] + y_vals[i:]
    
    y_std = [v for v in lit_dat[f'{feature_name} SD']]
    y_std = y_std[0:i] + [exp_dat[feature_name].std()] + y_std[i:]

    x_pos = i
    x_vals = [x_pos+1 + np.random.uniform(-.1, .1) for i in range(0, exp_dat.shape[0])]

    #ax.scatter(x_vals, exp_dat[feature_name], color='grey', alpha=.6, s=6)
    ax.scatter(x, y_vals, color='k')
    ax.errorbar(x, y_vals, y_std, capsize=2, ls='none', color='k')
    #ax.axhline(exp_dat[feature_name].mean(), color='grey', linestyle='--')

    feature_name_dict = {'MP': 'MP (mV)',
                         'APD90': r'$APD_{90}$ (ms)',
                         'dVdt': 'dV/dt (V/s)'}
    ax.set_ylabel(feature_name_dict[feature_name])

    ax.set_xticks([v for v in range(2, num_exps, 2)])
    [ax.axvline(v, color='k', alpha=.1) for v in range(1, num_exps)]
    ax.set_xlim(0, num_exps)
    #if 'dV' in feature_name:
    #    ax.set_ylim(0, 80)


def plot_dVdt(ax_small, ax_large, is_chamber_merged):
    feature_name = 'dVdt'
    lit_dat = pd.read_excel('./data/literature_ap_morph.xlsx', sheet_name='Sheet2', header=0)
    if is_chamber_merged:
        lit_dat = merge_chamber_dat(lit_dat)
    num_exps = lit_dat.shape[0] + 2
    exp_dat = pd.read_csv('./data/ap_features.csv')
    print(exp_dat.columns)

    lit_dat.sort_values('MP', inplace=True, ascending=False)

    x = np.linspace(1, lit_dat[feature_name].shape[0]+1, lit_dat[feature_name].shape[0]+1)

    i = 0
    exp_mean_mp = exp_dat['MP'].mean()
    for val in lit_dat['MP'].values:
        if val < exp_mean_mp:
            break
        i += 1

    y_vals = [v for v in lit_dat[feature_name].values]
    y_vals = y_vals[0:i] + [exp_dat[feature_name].mean()] + y_vals[i:]
    
    y_std = [v for v in lit_dat[f'{feature_name} SD']]
    y_std = y_std[0:i] + [exp_dat[feature_name].std()] + y_std[i:]

    x_pos = i
    x_vals = [x_pos+1 + np.random.uniform(-.1, .1) for i in range(0, exp_dat.shape[0])]








    for ax in [ax_small, ax_large]:
        #ax.scatter(x_vals, exp_dat[feature_name], color='grey', alpha=.6, s=6)
        ax.scatter(x, y_vals, color='k')
        ax.errorbar(x, y_vals, y_std, capsize=2, ls='none', color='k')
        #ax.axhline(exp_dat[feature_name].mean(), color='grey', linestyle='--')

        feature_name_dict = {'MP': 'MP (mV)',
                             'APD90': r'$APD_{90}$ (ms)',
                             'dVdt': 'dV/dt (V/s)'}

        ax.set_xticks([v for v in range(2, num_exps, 2)])
        [ax.axvline(v, color='k', alpha=.1) for v in range(1, num_exps)]
        ax.set_xlim(0, num_exps)

    ax_small.set_ylabel(feature_name_dict[feature_name])


def plot_literature_names(ax, is_chamber_merged):
    lit_dat = pd.read_excel('./data/literature_ap_morph.xlsx', sheet_name='Sheet2', header=0)
    if is_chamber_merged:
        lit_dat = merge_chamber_dat(lit_dat)

    lit_dat.sort_values('MP', inplace=True, ascending=False)
    exp_dat = pd.read_csv('./data/ap_features.csv')

    i = 0
    exp_mean_mp = exp_dat['MP'].mean()
    for val in lit_dat['MP'].values:
        if val < exp_mean_mp:
            break
        i += 1

    sorted_auths = lit_dat['First author'].values[0:i].tolist() + [r'Clark'] + lit_dat['First author'].values[i:].tolist()

    #ax.text(-0.26, .95, '1. Clark', fontsize=9)

    [ax.text(-.26, .95-.03*i, f'{i+1}. {a}', fontsize=9) for i, a in enumerate(sorted_auths)]

    #ax.text(0, 1, 'h1', fontsize=10)
    #ax.text(5, 5, 'h2')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def merge_chamber_dat(lit_dat):
    split_vals = [v.split('(')[0] for v in lit_dat['First author'].values]
    studies = list(set(split_vals))

    merged_dat = []

    for study in studies:
        curr_dat = lit_dat[lit_dat['First author'].str.contains(study)]

        if ('Herron' in study) or ('Lee' in study):
            merged_dat += curr_dat.values.tolist()
            continue

        if curr_dat.values.shape[0] == 1:
            merged_dat.append(curr_dat.values[0])
            continue

        #TODO
        #new_dat = [study[:-1],
        #           curr_dat['DOI'].iloc[0],
        #           curr_dat['Approach'].iloc[0],
        #           curr_dat['n'].sum(),
        #           np.array([])curr_dat['n'](),
        #           np.sqrt((curr_dat['MP SD']**2).sum()),
        #           curr_dat['APD90'].mean(),
        #           np.sqrt((curr_dat['APD90 SD']**2).sum()),
        #           curr_dat['dVdt'].mean(),
        #           np.sqrt((curr_dat['dVdt SD']**2).sum()),
        #           curr_dat['APA'].mean(),
        #           np.sqrt((curr_dat['APA SD']**2).sum()),
        #           ]
        #merged_dat.append(new_dat)

    new_df = pd.DataFrame(merged_dat, columns=lit_dat.columns)

    return new_df


def main():
    plot_figure(False)
    #plot_two_panel(False)


if __name__ == '__main__':
    main()
