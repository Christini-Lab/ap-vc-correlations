from os import listdir
import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import myokit

from utility_classes import VCSegment, VCProtocol

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)


def plot_figure():
    fig = plt.figure(figsize=(6.5, 6))
    fig.subplots_adjust(.1, .07, .95, .95)

    #grid = fig.add_gridspec(3, 2, hspace=.32, wspace=0.1, height_ratios=[3, 4, 4])
    grid = fig.add_gridspec(2, 2, hspace=.32, wspace=0.1, height_ratios=[4, 3])
    correlation_times = [600, 1262, 1986, 2760, 3641, 4300, 5840, 9040]


    ax_spont = fig.add_subplot(grid[1, 0])
    ax_flat = fig.add_subplot(grid[1, 1])

    cols = ['k', 'lightskyblue']

    plot_cc(ax_spont, 'spont', cols[0])
    plot_cc(ax_flat, 'flat', cols[1])

    vc_subgrid = grid[0,:].subgridspec(2, 1, wspace=.9, hspace=.1, height_ratios=[2,3])
    ax_vc_v = fig.add_subplot(vc_subgrid[0])
    ax_vc_i = fig.add_subplot(vc_subgrid[1])

    windows = []
    zoom_axes = []

    plot_vc(ax_vc_v, ax_vc_i, windows, zoom_axes, cols=cols)
    axs = [ax_spont, ax_flat, ax_vc_v, ax_vc_i]

    [ax_vc_v.axvline(curr_t, color='grey', linestyle='--', alpha=.5) for curr_t in correlation_times]
    seg_names = [r'$I_{6mV}$', r'$I_{Kr}$', r'$I_{CaL}$', r'$I_{Na}$', '$I_{to}$', '$I_{K1}$', '$I_{f}$', '$I_{Ks}$']

    for i, curr_t in enumerate(correlation_times):
        if i == 0:
            ax_vc_v.text(curr_t-500, 70, seg_names[i])
        else:
            ax_vc_v.text(curr_t-20, 70, seg_names[i])

    ax_spont.arrow(220, 35, 0, -27, head_width=6, head_length=3, color='k')
    ax_vc_i.arrow(5500, 5, 0, -7, head_width=100, head_length=1, color='k')


    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    letters = ['A', 'B', 'C', 'D', 'E']
    #for i, ax in enumerate([ax_spont, ax_flat, ax_vc_v, ax_vc_v1, ax_vc_v2]):
    for i, ax in enumerate([ax_vc_v, ax_spont, ax_flat]):
        if i == 2:
            ax.set_title(letters[i], y=.98, x=-.05)
        else:
            ax.set_title(letters[i], y=.98, x=-.1)


    ax_spont.set_ylabel('Voltage (mV)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-ap_vc_hetero.pdf', transparent=True)

    plt.show()


def plot_figure_only_ap():
    fig = plt.figure(figsize=(4.5, 4))
    fig.subplots_adjust(.15, .12, .95, .95)

    grid = fig.add_gridspec(1, 1)

    ax_spont = fig.add_subplot(grid[0])

    plot_cc(ax_spont, 'flat', 'k')
    plot_cc(ax_spont, 'spont', 'k')

    ax_spont.spines['top'].set_visible(False)
    ax_spont.spines['right'].set_visible(False)

    ax_spont.set_ylabel('Voltage (mV)')

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig('./figure-pdfs/f-ap_my_dat_hetero.pdf', transparent=True)

    plt.show()


def plot_cc(ax, cc_type, col=None):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 250, 2500)

    n = 0

    for f in all_files:
        if '.DS' in f:
            continue

        ap_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')
        ap_dat, cc_shape = get_ap(ap_dat)

        if cc_shape == cc_type:
            if cc_type == 'spont':
                if col is None:
                    ax.plot(t, ap_dat, '#377eb8', alpha = .3, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = .3, rasterized=True)
                t_sp = t
                ap_dat_sp = ap_dat
            else:
                if col is None:
                    ax.plot(t, ap_dat, '#ff7f00', alpha = .5, rasterized=True)
                else:
                    ax.plot(t, ap_dat, col, alpha = .5, rasterized=True)

            if f == '6_033021_4_alex_control':
                ax.plot(t, ap_dat, 'pink', rasterized=True)
            #if f == '4_022421_1_alex_cisapride':
            #    ax.plot(t, ap_dat, 'pink', rasterized=True)


            #ax.plot(ap_dat, 'k', alpha = .3)
            n += 1
        else:
            continue

    #if cc_type == 'spont':
    #    ax.plot(t_sp, ap_dat_sp, 'pink')

    ax.text(100, 40, f'n={n}')
    ax.set_ylim(-70, 50)
    ax.set_xlabel('Time (ms)')


def plot_vc(ax_vc_v, ax_vc_i, windows, zoom_axes, cols):
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    t = np.linspace(0, 250, 2500)

    n = 0

    st = 40000
    end = 140000

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        exp_dat = pd.read_csv('./data/ap_features.csv')

        t = vc_dat['Time (s)'][st:end]*1000-4000

        if np.isnan(exp_dat[exp_dat['File'] == f]['CL'].values[0]):
            #FLAT
            col = '#ff7f00'
            col = cols[1]
            alpha=.25
        else:
            #SPONT
            col = '#377eb8'
            col = cols[0]
            alpha=.1


        ax_vc_i.plot(t, vc_dat['Current (pA/pF)'][st:end], color=col, alpha=alpha, rasterized=True)

        if f == '6_033021_4_alex_control':
            ax_vc_i.plot(t, vc_dat['Current (pA/pF)'][st:end], 'pink', linewidth=1.5, rasterized=True)
        #if f == '4_022421_1_alex_cisapride':
        #    ax_vc_i.plot(t, vc_dat['Current (pA/pF)'][st:end], 'pink', linewidth=1.5, rasterized=True)
            
        
        for i, w in enumerate(windows):
            z_st = st+w[0]*10
            z_end = st+w[1]*10
            t_z = vc_dat['Time (s)'][z_st:z_end]*1000-4000
            c_z = vc_dat['Current (pA/pF)'][z_st:z_end]
            zoom_axes[i][1].plot(t_z, c_z, color=col, alpha=.1)

    #ax_vc_i.plot(t, vc_dat['Current (pA/pF)'][st:end], color='pink')

    ax_vc_v.plot(t, vc_dat['Voltage (V)'][st:end]*1000, 'k')
    for i, w in enumerate(windows):
        z_st = st+w[0]*10
        z_end = st+w[1]*10
        t_z = vc_dat['Time (s)'][z_st:z_end]*1000-4000
        v_z = vc_dat['Voltage (V)'][z_st:z_end]*1000

        zoom_axes[i][0].plot(t_z, v_z, 'k')

    letters = ['D', 'E']

    for i, w in enumerate(windows):
        ax_vc_v.axvspan(w[0], w[1], color='#999999', alpha=.3)
        zoom_axes[i][0].axvspan(w[0], w[1], color='#999999', alpha=.2)
        ax_vc_v.text((w[0]+w[1])/2-15, 25, letters[i], fontsize=8)

    ax_vc_v.tick_params(labelbottom=False)
    ax_vc_i.set_ylim(-15, 15)

    ax_vc_v.set_ylabel('Voltage (mV)')
    ax_vc_i.set_ylabel('Current (pA/pF)')
    ax_vc_i.set_xlabel('Time (ms)')

    if zoom_axes:
        zoom_axes[0][1].set_ylim(-15, 15)
        zoom_axes[1][1].set_ylim(-20, 0)

        zoom_axes[0][0].set_ylabel('Voltage (mV)')
        zoom_axes[0][1].set_ylabel('Current (pA/pF)')

        zoom_axes[0][1].set_xlabel('Time (ms)')
        zoom_axes[1][1].set_xlabel('Time (ms)')


def get_ap(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    if (((v.max() - v.min()) < 20) or (v.max() < 0)):
        return v[60000:62500], 'flat'

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')

    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return v[0:2500], 'flat'
        #import pdb
        #pdb.set_trace()
        #return 

    idx_start = peak_idxs[0] - 50
    idx_end = idx_start + 2500
    #min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    #min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    #search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    #amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    #v_90 = min_v + amplitude * .1
    #idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    return v[idx_start:idx_end], 'spont'


def plot_mods(ax, windows, zoom_axes):
    cm = 30
    kernik_times, kernik_dat = get_mod_response('kernik', cm=cm)
    paci_times, paci_dat = get_mod_response('paci', cm=cm)

    i_out = [v / cm for v in kernik_dat['voltageclamp.Iout']] 
    ax.plot(kernik_dat['engine.time'], i_out, label='Kernik', color='k', linestyle='--', linewidth=1.3)

    for i, w in enumerate(windows):
        st = np.argmin(np.abs(kernik_times - w[0]))
        end = np.argmin(np.abs(kernik_times - w[1]))

        z_t = kernik_dat['engine.time'][st:end]
        z_c = i_out[st:end]

        zoom_axes[i][1].plot(z_t, z_c, label='Kernik', color='k', linestyle='--', linewidth=1.3)

    i_out = [v / cm for v in paci_dat['voltageclamp.Iout']] 
    #ax.plot(paci_dat['engine.time'], i_out, label='Paci', color='k', linestyle='dotted', linewidth=1.3)

    for i, w in enumerate(windows):
        st = np.argmin(np.abs(paci_times - w[0]))
        end = np.argmin(np.abs(paci_times - w[1]))

        z_t = paci_dat['engine.time'][st:end]
        z_c = i_out[st:end]

        zoom_axes[i][1].plot(z_t, z_c, label='Paci', color='k', linestyle='dotted', linewidth=1.3)

    #ax.legend(loc='upper right', framealpha=1)


def plot_kernik_cc(ax):
    mod, proto, x = myokit.load('mmt/kernik_2019_mc.mmt')
    sim = myokit.Simulation(mod, proto)
    t = 10000 
    times = np.arange(0, t, .1)
    dat = sim.run(t, log_times=times)

    t = np.array([t for t in dat['engine.time']])
    v = dat['membrane.V']

    ax.plot(t[4950:]-505, v[4950:], 'k--', label='iPSC-CM Model')
    ax.legend()


#UTILITY
def return_vc_proto(scale=1):
    segments = [
            VCSegment(757, 6),
            VCSegment(7, -41),
            VCSegment(101, 8.5),
            VCSegment(500, -80),
            VCSegment(106, -81),
            VCSegment(103, -2, -34),
            VCSegment(500, -80),
            VCSegment(183, -87),
            VCSegment(102, -52, 14),
            VCSegment(500, -80),
            VCSegment(272, 54, -107),
            VCSegment(103, 60),
            VCSegment(500, -80),
            VCSegment(52, -76, -80),
            VCSegment(103, -120),
            VCSegment(500, -80),
            VCSegment(940, -120),
            VCSegment(94, -77),
            VCSegment(8.1, -118),
            VCSegment(500, -80),
            VCSegment(729, 55),
            VCSegment(1000, 48),
            VCSegment(895, 59, 28),
            VCSegment(900, -80)
            ]

    new_segments = []
    for seg in segments:
        if seg.end_voltage is None:
            new_segments.append(VCSegment(seg.duration*scale, seg.start_voltage*scale))
        else:
            new_segments.append(VCSegment(seg.duration*scale,
                                          seg.start_voltage*scale,
                                          seg.end_voltage*scale))

    return VCProtocol(new_segments)


def get_mod_response(mod_name, cm):
    if mod_name == 'kernik':
        mod = myokit.load_model('./mmt/kernik_artifact_fixed.mmt')
        mod['geom']['Cm'].set_rhs(cm)
    if mod_name == 'paci':
        mod = myokit.load_model('./mmt/paci_artifact_ms_fixed.mmt')
        mod['cell']['Cm'].set_rhs(cm)

    mod['voltageclamp']['cm_est'].set_rhs(cm)

    p = mod.get('engine.pace')
    p.set_binding(None)

    vc_proto = return_vc_proto()

    proto = myokit.Protocol()
    proto.add_step(-80, 10000)

    piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

    ####
    p = mod.get('engine.pace')
    p.set_binding(None)

    new_seg_dict = {}
    for k, vol in segment_dict.items():
        new_seg_dict[k] = vol

    segment_dict = new_seg_dict

    mem = mod.get('voltageclamp')

    for v_name, st in segment_dict.items():
        v_new = mem.add_variable(v_name)
        v_new.set_rhs(st)

    vp = mem.add_variable('vp')
    vp.set_rhs(0)

    v_cmd = mod.get('voltageclamp.Vc')
    v_cmd.set_binding(None)
    vp.set_binding('pace')

    v_cmd.set_rhs(piecewise)
    ####

    #t = proto.characteristic_time()
    sim = myokit.Simulation(mod, proto)

    times = np.arange(0, t_max, 0.1)

    dat = sim.run(t_max, log_times=times)

    return times, dat


def plot_proto_response():
    kernik_times, kernik_dat = get_mod_response('kernik')
    paci_times, paci_dat = get_mod_response('paci')

    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(12, 8))

    axs[0].plot(kernik_dat['engine.time'], kernik_dat['membrane.V'], 'k')

    #I_ion
    axs[1].plot(kernik_dat['engine.time'], kernik_dat['voltageclamp.Iout'], label='Kernik')
    axs[1].plot(np.array(paci_dat['engine.time'])*1000, np.array(paci_dat['voltageclamp.Iout']), label='Paci')

    #INaCa
    axs[2].plot(kernik_dat['engine.time'], kernik_dat['voltageclamp.ILeak'], label='Kernik')
    axs[2].plot(np.array(paci_dat['engine.time'])*1000, np.array(paci_dat['voltageclamp.ILeak']), label='Paci')

    axs[1].axhline(y=0, color='grey')
    axs[2].axhline(y=0, color='grey')


    fs = 14
    axs[0].set_ylabel('mV', fontsize=fs)
    axs[1].set_ylabel(r'$I_m$ (pA/pF)', fontsize=fs)
    axs[2].set_ylabel(r'$I_{NaCa}$ (pA/pF)', fontsize=fs)
    axs[3].set_ylabel(r'$Ca_i$', fontsize=fs)
    axs[4].set_ylabel(r'$Na_i$', fontsize=fs)
    axs[1].set_xlabel('Time (ms)', fontsize=fs)

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=fs-4)

    plt.legend()

    plt.show()


def main():
    plot_figure()


if __name__ == '__main__':
    main()

