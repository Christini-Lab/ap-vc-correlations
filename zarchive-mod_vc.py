from os import listdir, mkdir
from multiprocessing import Pool
import random
import scipy as sp

import myokit
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utility_classes import VCSegment, VCProtocol

plt.rcParams['lines.linewidth'] = .9
plt.rcParams['lines.markersize'] = 4
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rc('legend', fontsize = 8)



global MOD_Cm 
MOD_Cm = 45 #pF
global MOD_rseries
MOD_rseries = .02 #Gohms
global MOD_gLeak
MOD_gLeak = .5 

global LEAK_NAME
LEAK_NAME = ['gLeak']
#LEAK_NAME = []
global CM_NAMES
CM_NAMES = ['cm', 'cm_est'] 
#CM_NAMES = []

global RSERIES_NAMES 
RSERIES_NAMES = ['rseries', 'rseries_est']
#RSERIES_NAMES = []

global G_NAMES
G_NAMES = ['gNa', 'gKr', 'gCaL', 'gKs', 'gK1', 'gf',
                    'gto', 'gNaK', 'gNaCa', 'gbNa', 'gbCa']
#G_NAMES = []

global ALL_PARAMS
ALL_PARAMS = LEAK_NAME + CM_NAMES + RSERIES_NAMES + G_NAMES

global IS_ARTIFACT
IS_ARTIFACT = True


def plot_ical_kernik():
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5))

    styles = ['k', 'r--']
    labs = ['Baseline', r'$I_{CaL}$ block']

    for i, g_scale in enumerate([100, 110]):
        model_path = './mmt/kernik_artifact_fixed.mmt'
        model_path = './mmt/kernik_artifact.mmt'
        mod = myokit.load_model(model_path)
        mod['geom']['Cm'].set_rhs(MOD_Cm)

        mod['voltageclamp']['cm_est'].set_rhs(MOD_Cm)
        mod['voltageclamp']['rseries'].set_rhs(MOD_rseries)
        mod['voltageclamp']['rseries_est'].set_rhs(MOD_rseries)
        mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)

        #ki.Ki      =  1.047488243941121e+02

        #if i == 1:
        #    mod['ki']['Ki'].set_rhs(g_scale)

        p = mod.get('engine.pace')
        p.set_binding(None)

        prestep = 10000
        vc_proto = return_vc_proto(prestep_size=prestep)

        proto = myokit.Protocol()
        proto.add_step(-80, prestep+10000)

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

        sim = myokit.Simulation(mod, proto)

        times = np.arange(0, t_max, 0.1)

        sim.set_max_step_size(1)

        dat = sim.run(t_max, log_times=times)

        cm = mod['voltageclamp']['cm_est'].value()
        t = dat.time()
        i_out = [v / cm for v in dat['voltageclamp.Iout']]
        v = dat['voltageclamp.Vc']
        axs[1].plot(t, i_out, styles[i], label=labs[i])

    axs[0].plot(t, v, 'k')

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel(r'$V_m$ (mV)')
    axs[1].set_ylabel(r'$I_{out}$ (A/F)')
    axs[-1].set_xlabel('Time (ms)')

    axs[1].legend()
    plt.show()


def run_mod_pop_models(mod_name, pop_size=4):
    population_dirs = listdir('./data/mod_populations/')
    population_dirs = [v for v in population_dirs if 'pop_' in v]

    if not population_dirs:
        curr_dir = f'./data/mod_populations/pop_1_{mod_name}'
    else:
        dir_nums = [int(v.split('_')[1]) for v in population_dirs]
        curr_dir = f'./data/mod_populations/pop_{np.max(dir_nums)+1}_{mod_name}'

    #scale_range = [1/2.0, 2.0]
    scale_range = [1/4.0, 4.0]

    mkdir(curr_dir)
    mkdir(f'{curr_dir}/cc')
    mkdir(f'{curr_dir}/vc')
    mkdir(f'{curr_dir}/params')

    f = open(f'{curr_dir}/meta.txt', 'a')
    f.write(f'Model name: {mod_name}\n')
    f.write(f'Scale range: {scale_range}\n')
    f.write(f'Population size: {pop_size}\n')
    f.write(f'Parameters: \n')
    [f.write(f'\t{p}\n') for p in ALL_PARAMS]
    f.write(f'GLeak: {MOD_gLeak}\n')
    f.close()

    mapped_inputs = [[mod_name, scale_range]] * pop_size
    #scale_inputs = [scale_range] * pop_size 
    #mod_names = [mod_name] * pop_size

    p = Pool()
    p.map(get_save_individual, mapped_inputs)#, np.array(mod_names),
                             #      np.array(scale_inputs)[:, 0],
                             #      np.array(scale_inputs)[:, 1])
    #vals = list(map(get_save_individual, mapped_inputs))

    f = open(f'{curr_dir}/meta.txt', 'a')
    if IS_ARTIFACT:
        f.write(f'Model: {f} Artifact\n')
    else:
        f.write(f'Model: {f} Leak\n')
    f.write(f'COMPLETED!')
    f.close()


def get_save_individual(mapped_inputs):
    sp.random.seed()
    mod_name = mapped_inputs[0]
    scale_range = mapped_inputs[1]
    leak_name = LEAK_NAME # can be [] or ['gLeak']
    leak_val = [get_random_log_uniform(scale_range)]

    cm_names = CM_NAMES 
    cm_vals = [get_random_log_uniform(scale_range)] * 2

    rseries_names = RSERIES_NAMES 
    rseries_vals = [get_random_log_uniform(scale_range)] * 2

    g_names = G_NAMES 
    g_vals = [get_random_log_uniform(scale_range) for v in g_names]
 
    all_cc_names = leak_name + g_names
    all_cc_values = leak_val + g_vals 
    all_cc_params = dict(zip(all_cc_names, all_cc_values))

    all_vc_names = leak_name + cm_names + rseries_names + g_names
    all_vc_values = leak_val + cm_vals + rseries_vals + g_vals 
    all_vc_params = dict(zip(all_vc_names, all_vc_values))

    all_cc_params = get_mod_params(mod_name, all_cc_params)
    all_vc_params = get_mod_params(mod_name, all_vc_params)

    t_cc, v_cc = get_cc_response(mod_name, all_cc_params)
    try:
        if IS_ARTIFACT:
            t_vc, i_vc, v_vc = get_vc_artifact_response(mod_name, all_vc_params)
        else:
            t_vc, i_vc, v_vc = get_vc_leak_response(mod_name, all_vc_params)
    except:
        return

    cc_pd = pd.DataFrame({'t': t_cc, 'V': v_cc})
    vc_pd = pd.DataFrame({'t': t_vc, 'V': v_vc, 'i_out': i_vc})

    population_dirs = listdir('./data/mod_populations/')
    population_dirs = [v for v in population_dirs if 'pop_' in v]
    dir_nums = [int(v.split('_')[1]) for v in population_dirs]
    curr_dir = f'./data/mod_populations/pop_{np.max(dir_nums)}_{mod_name}'

    #write cc, vc, and params
    rand_hash = random.getrandbits(64)

    cc_pd.to_csv(f'{curr_dir}/cc/cc_{rand_hash}.csv', index=False)
    vc_pd.to_csv(f'{curr_dir}/vc/vc_{rand_hash}.csv', index=False)
    f = open(f'{curr_dir}/params/params_{rand_hash}.csv', 'a')
    [f.write(f'{k},') for k, v in all_vc_params.items()]
    f.write('\n')
    [f.write(f'{v},') for k, v in all_vc_params.items()]
    f.close()

    print('Made it!')


def get_cc_response(mod_name, all_params, with_all_dat=False):
    if mod_name == 'Kernik':
        model_path = './mmt/kernik_leak_fixed.mmt'
        mod, p, x = myokit.load(model_path)
        mod['geom']['Cm'].set_rhs(MOD_Cm)
    if mod_name == 'Paci':
        model_path = './mmt/paci_leak_ms_fixed.mmt'
        mod, p, x = myokit.load(model_path)
        mod['cell']['Cm'].set_rhs(MOD_Cm) 
    
    mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)

    for k, scale in all_params.items():
        group, name = k.split('.')
        model_value = mod[group][name].value()
        mod[group][name].set_rhs(model_value * scale)

    sim = myokit.Simulation(mod)
    prepace = 100000
    sim.pre(prepace)

    dat = sim.run(10000, log_times=np.arange(0, 10000, 1))

    t = dat.time()
    v = dat['membrane.V']

    if with_all_dat:
        return t, dat
    else:
        return t, v


def get_vc_artifact_response(mod_name, all_params, with_all_dat=False):
    if mod_name == 'Kernik':
        model_path = './mmt/kernik_artifact_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['geom']['Cm'].set_rhs(MOD_Cm)
    if mod_name == 'Paci':
        model_path = './mmt/paci_artifact_ms_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['cell']['Cm'].set_rhs(MOD_Cm) 
    
    mod['voltageclamp']['cm_est'].set_rhs(MOD_Cm)
    mod['voltageclamp']['rseries'].set_rhs(MOD_rseries)
    mod['voltageclamp']['rseries_est'].set_rhs(MOD_rseries)
    mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)

    for k, scale in all_params.items():
        group, name = k.split('.')
        model_value = mod[group][name].value()
        mod[group][name].set_rhs(model_value * scale)


    p = mod.get('engine.pace')
    p.set_binding(None)

    prestep = 50000
    vc_proto = return_vc_proto(prestep_size=prestep)

    proto = myokit.Protocol()
    proto.add_step(-80, prestep+10000)

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

    sim = myokit.Simulation(mod, proto)

    times = np.arange(0, t_max, 0.1)

    sim.set_max_step_size(1)

    dat = sim.run(t_max, log_times=times)

    cm = mod['voltageclamp']['cm_est'].value()
    t = dat.time()
    i_out = [v / cm for v in dat['voltageclamp.Iout']]
    v = dat['voltageclamp.Vc']

    if with_all_dat:
        return times, i_out, v, dat
    else:
        return times, i_out, v


def get_vc_leak_response(mod_name, all_params, with_all_dat=False):
    if mod_name == 'Kernik':
        model_path = './mmt/kernik_leak_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['geom']['Cm'].set_rhs(MOD_Cm)
    if mod_name == 'Paci':
        model_path = './mmt/paci_leak_ms_fixed.mmt'
        mod = myokit.load_model(model_path)
        mod['cell']['Cm'].set_rhs(MOD_Cm) 
    
    mod['voltageclamp']['gLeak'].set_rhs(MOD_gLeak)

    for k, scale in all_params.items():
        group, name = k.split('.')
        model_value = mod[group][name].value()
        mod[group][name].set_rhs(model_value * scale)

    prestep = 50000
    vc_proto = return_vc_proto(prestep_size=prestep)

    proto = myokit.Protocol()
    proto.add_step(-80, prestep+10000)

    piecewise, segment_dict, t_max = vc_proto.get_myokit_protocol()

    new_seg_dict = {}
    for k, vol in segment_dict.items():
        new_seg_dict[k] = vol

    segment_dict = new_seg_dict

    p = mod.get('engine.pace')
    p.set_binding(None)
    
    v = mod.get('membrane.V')
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace') # Bind to the pacing mechanism

    mem = mod.get('membrane')
    v = mem.get('V')
    v.set_binding(None)

    for v_name, st in segment_dict.items():
        v_new = mem.add_variable(v_name)
        v_new.set_rhs(st)

    vp = mem.add_variable('vp')
    vp.set_rhs(0)
    vp.set_binding('pace')

    v.set_rhs(piecewise)

    sim = myokit.Simulation(mod, proto)
    sim.set_max_step_size(1)
    times = np.arange(0, t_max, 0.1)

    dat = sim.run(t_max, log_times=times)

    t = dat.time()
    i_out = dat['membrane.i_ion']
    v = dat['membrane.V']

    if with_all_dat:
        return times, i_out, v, dat
    else:
        return times, i_out, v


def get_mod_individual(mod_name, params, param_range):
    pass


def get_mod_params(mod_name, param_dict):
    """
    Artifact Parameters:
        Cm: 
            Kernik is geom.Cm
            Paci is cell.Cm
        cm_est
            voltageclamp.cm_est
        rseries
            voltageclamp.rseries
        rseries_est
            voltageclamp.rseries_est
        gLeak
            voltageclamp.gLeak

    Conductances
        gNa
            Kernik: ina.g_Na
            Paci: ina.g
        gKr
            Kernik: ikr.g_Kr
            Paci: ikr.g
        gCaL
            Kernik: ical.g_scale
            Paci: ical.P_CaL
        gKs
            Kernik: iks.g_scale
            Paci: iks.g
        gK1
            Kernik: ik1.g_K1
            Paci: ik1.g
        gf
            Kernik: ifunny.g_f
            Paci: if.g
        gto
            Kernik: ito.g_to
            Paci: ito.g

        CaT
            Kernik: icat.g_CaT
            Paci: NONE
        NaK
            Kernik: inak.g_scale
            Paci: inak.PNaK
        NaCa
            Kernik: inaca.g_scale
            Paci: inaca.kNaCa
        bNa
            Kernik: ibna.g_b_Na
            Paci: ibna.g
        bCa
            Kernik: ibca.g_b_Ca
            Paci: ibca.g
    """
    all_params = ['cm', 'cm_est', 'rseries', 'rseries_est', 'gLeak', 'gNa', 'gKr', 'gCaL', 'gKs', 'gK1', 'gf', 'gto', 'gNaK', 'gNaCa', 'gbNa', 'gbCa']

    artifact_params = ['geom.Cm', 'voltageclamp.cm_est',
                       'voltageclamp.rseries',
                       'voltageclamp.rseries_est',
                       'voltageclamp.gLeak']
    kernik_params = artifact_params + [
                     'ina.g_Na', 'ikr.g_Kr', 'ical.g_scale',
                     'iks.g_scale', 'ik1.g_K1', 'ifunny.g_f',
                     'ito.g_to', 'inak.g_scale', 'inaca.g_scale',
                     'ibna.g_b_Na', 'ibca.g_b_Ca']

    artifact_params = ['cell.Cm', 'voltageclamp.cm_est',
                       'voltageclamp.rseries',
                       'voltageclamp.rseries_est',
                       'voltageclamp.gLeak']
    paci_params = artifact_params + [
                   'ina.g', 'ikr.g', 'ical.P_CaL', 'iks.g', 'ik1.g',
                   'if.g', 'ito.g', 'inak.PNaK', 'inaca.kNaCa',
                   'ibna.g', 'ibca.g']

    
    if mod_name == 'Kernik':
        key = dict(zip(all_params, kernik_params)) 
    else:
        key = dict(zip(all_params, paci_params))

    #[print(f'{k}: {v}') for k, v in kernik_key.items()]
    #[print(f'{k}: {v}') for k, v in paci_key.items()]

    #params = [p for p in param_dict.keys()]

    mod_params = []
    
    [mod_params.append(key[k]) for k, v in param_dict.items()]

    param_vals = [v for k, v in param_dict.items()]

    return dict(zip(mod_params, param_vals))


def return_vc_proto(scale=1, prestep_size=0):
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

    if prestep_size != 0:
        segments = [VCSegment(prestep_size, -80)] + segments

    new_segments = []
    for seg in segments:
        if seg.end_voltage is None:
            new_segments.append(VCSegment(seg.duration*scale, seg.start_voltage*scale))
        else:
            new_segments.append(VCSegment(seg.duration*scale,
                                          seg.start_voltage*scale,
                                          seg.end_voltage*scale))

    return VCProtocol(new_segments)


plot_ical_kernik()
