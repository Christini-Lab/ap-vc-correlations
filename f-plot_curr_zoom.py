from os import listdir
import matplotlib
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


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
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(6.5, 6))
    fig.subplots_adjust(.1, .1, .95, .95)

    curr = 'I_Ks'
    feature = 'APD90'

    plot_vc(axs, current=curr, feature=feature)
    plot_mods(axs, curr)

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.savefig(f'./figure-pdfs/f-ap_vc_hetero_{curr}.pdf', transparent=True)

    axs[0].set_ylabel(r'$V_m$ (mV)')
    axs[1].set_ylabel(r'$I_{out}$ (A/F)')
    axs[-1].set_xlabel('Time (ms)')

    plt.show()


def plot_vc(axs, current, feature='MP'):
    all_files = listdir('./data/cells')

    all_currs = []
    all_ap_features = pd.read_csv('./data/ap_features.csv')


    t = np.linspace(0, 250, 2500)

    correlation_times = [600, 1263, 1986, 2760, 3641, 4300, 5840, 9040, 9080]
    current_times = dict(zip(['I_6mV', 'I_Kr', 'I_CaL', 'I_Na', 'I_to', 'I_K1', 'I_f', 'I_Ks'], correlation_times))

    mid = 40000 + current_times[current] * 10
    st = mid - 400
    end = mid + 400

    all_currs_dict = {}

    for f in all_files:
        if '.DS' in f:
            continue

        vc_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_vcp_70_70.csv')
        exp_dat = pd.read_csv('./data/ap_features.csv')

        t = (vc_dat['Time (s)'])*1000 - 4000

        if np.isnan(exp_dat[exp_dat['File'] == f]['CL'].values[0]):
            #FLAT
            col = '#ff7f00'
        else:
            #SPONT
            col = '#377eb8'

        axs[1].plot(t[st:end], vc_dat['Current (pA/pF)'][st:end], color=col, alpha=.3)

        all_currs_dict[f] = vc_dat['Current (pA/pF)'][st:end]

    axs[0].plot(t[st:end], vc_dat['Voltage (V)'][st:end]*1000, 'k')

    current_limits = [[-7, 7], [-7, 7], [-50, 7], [-7, 7], [-7, 7], [-12, 2], [-15, 0], [-15, 10]]
    current_lims = dict(zip(['I_6mV', 'I_Kr', 'I_CaL', 'I_Na', 'I_to', 'I_K1', 'I_f', 'I_Ks', 'I_last'], current_limits))

    axs[1].set_ylim(current_lims[current][0], current_lims[current][1])
    for i in range(0, 5):
        axs[i].axvline(current_times[current], color='grey', linestyle='--', linewidth=1.4)

    all_currs = np.array(all_currs)

    all_currs_df = pd.DataFrame(all_currs_dict)
    all_currs_df = all_currs_df.reindex(
            sorted(all_currs_df.columns), axis=1)
    all_ap_features = all_ap_features.sort_values('File')
    all_currs = all_currs_df.values

    #ap_feature_key = {'MP': 0, 'APD90': 1, 'dVdt': 3}
    #feature_idx = ap_feature_key[feature]

    all_rows = np.invert(np.isnan(all_ap_features[feature].astype(float)))
    all_corr = np.zeros(all_currs.shape[0])
    all_pvals = np.zeros(all_currs.shape[0])

    for i in range(0, all_currs.shape[0]):
        corr_dat = np.corrcoef(all_ap_features[feature][all_rows].astype('float64').values, all_currs[i, all_rows].astype('float64'))
        all_corr[i] = corr_dat[0][1]
        all_pvals[i] = ttest_ind(all_ap_features[feature][all_rows].astype('float64'), all_currs[i, all_rows].astype('float64'))[1]

    axs[2].plot(t[st:end], all_corr, color='k', label='Correlation coefficient')
    axs[2].set_ylabel(f'r (AP {feature}'+r', $I_{out}$)')
    axs[2].legend()


def plot_mods(axs, current):
    cm = 30
    kernik_times, kernik_dat = get_mod_response('kernik', cm=cm)
    paci_times, paci_dat = get_mod_response('paci', cm=cm)

    i_out = [v / cm for v in kernik_dat['voltageclamp.Iout']]
    #axs[2].plot(kernik_dat['engine.time'], i_out, label='Kernik',
    #                        color='k', linestyle='--', linewidth=1.3)

    correlation_times = [600, 1263, 1986, 2760, 3641, 4300, 5840, 9040]
    current_times = dict(zip(['I_6mV', 'I_Kr', 'I_CaL', 'I_Na', 'I_to', 'I_K1', 'I_f', 'I_Ks'], correlation_times))

    mid = current_times[current]
    w = [mid - 40, mid + 40]

    st = np.argmin(np.abs(kernik_times - w[0]))
    end = np.argmin(np.abs(kernik_times - w[1]))

    z_t = kernik_dat['engine.time'][st:end]
    z_c = i_out[st:end]

    axs[1].plot(z_t, z_c, label='Kernik', color='k', linestyle='--', linewidth=1.3)

    plot_percentages(axs, kernik_times, kernik_dat, cm, w, mid, mod_name='Kernik')

    i_out = [v / cm for v in paci_dat['voltageclamp.Iout']]
    z_t = kernik_dat['engine.time'][st:end]
    z_c = i_out[st:end]
    axs[1].plot(z_t, z_c, label='Paci', color='k', linestyle='dotted', linewidth=1.3)
    axs[0].plot(z_t, paci_dat['voltageclamp.Vc'][st:end], label='Paci', color='k', linestyle='dotted', linewidth=1.3)

    plot_percentages(axs, paci_times, paci_dat, cm, w, mid, mod_name='Paci')

    axs[1].legend(loc='upper right', framealpha=1)


def plot_percentages(axs, times, dat, cm, w, corr_time, mod_name):
    if mod_name == 'Kernik':
        all_curr_names = ['ik1.i_K1', 'ito.i_to', 'ikr.i_Kr', 'iks.i_Ks', 'ical.i_CaL', 'icat.i_CaT', 'inak.i_NaK', 'ina.i_Na', 'inaca.i_NaCa', 'ipca.i_PCa', 'ifunny.i_f', 'ibna.i_b_Na', 'ibca.i_b_Ca']
    if mod_name == 'Paci':
        all_curr_names = ['ik1.IK1', 'ikr.IKr', 'iks.IKs', 'ito.Ito', 'if.If', 'ina.INa', 'ibna.IbNa', 'ical.ICaL', 'ipca.IpCa', 'ibca.IbCa', 'inak.INaK', 'inaca.INaCa']
    leak_name = 'voltageclamp.ILeak'

    st = np.argmin(np.abs(times - w[0]))
    end = np.argmin(np.abs(times - w[1]))

    current_dict = {}

    for curr in all_curr_names:
        curr_current = dat[curr][st:end]
        current_dict[curr] = [c for c in curr_current]

    current_dict[leak_name] = [i / cm for i in dat[leak_name][st:end]]

    all_curr_dat = pd.DataFrame(current_dict)

    all_curr_dat['i_abs_sum'] = all_curr_dat.abs().sum(1) 
    curr_dat_percentages = all_curr_dat.abs().div(all_curr_dat.i_abs_sum, 0).copy()

    times = times[st:end]
    min_arg = np.abs(times - corr_time).argmin()

    if mod_name == 'Kernik':
        ls = 'dashed'
        ax_num = 3
    else:
        ls = 'dotted'
        ax_num = 4

    i_other = np.zeros(all_curr_dat.shape[0])
    pct_other = np.zeros(all_curr_dat.shape[0])

    for col_name in curr_dat_percentages.columns:
        if col_name == 'i_abs_sum':
            continue

        if curr_dat_percentages[col_name].iloc[min_arg] > .15:
            #axs[2].plot(times,
            #        all_curr_dat[col_name], label=col_name)
            axs[ax_num].plot(times,
                    curr_dat_percentages[col_name],
                    label=col_name)
        else:
            i_other += all_curr_dat[col_name]
            pct_other += curr_dat_percentages[col_name]

    axs[ax_num].plot(times,
            pct_other, label=f'All other',
            color='k', linestyle='--')
    axs[ax_num].legend()
    axs[3].set_ylabel(r'KC %$I_x$')
    axs[4].set_ylabel(r'Paci %$I_x$')

    axs[2].legend()
        

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


def return_vc_proto(scale=1):
    segments = [
            #IKr
            VCSegment(756.9, 6),
            VCSegment(7, -41.3),
            VCSegment(101, 8.5),
            #ICaL
            VCSegment(500, -80),
            VCSegment(106.1, -81),
            VCSegment(103, -2, -34),
            #INa
            VCSegment(500.7, -80),
            VCSegment(183, -87),
            VCSegment(102, -52, 14),
            #Ito
            VCSegment(500, -80),
            VCSegment(272, 54, -107),
            VCSegment(103, 60),
            #IK1
            VCSegment(500, -80),
            VCSegment(52, -76, -80),
            VCSegment(103, -120),
            #If
            VCSegment(500, -80),
            VCSegment(936.6, -120),
            VCSegment(94, -77),
            VCSegment(8.1, -118),
            #IKs
            VCSegment(500, -80),
            VCSegment(729, 55),
            VCSegment(1000, 48),
            VCSegment(892.6, 59, 28),
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



def main():
    plot_figure()


if __name__ == '__main__':
    main()
