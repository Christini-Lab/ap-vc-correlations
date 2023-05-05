import pandas as pd
from os import listdir
import numpy as np
from scipy.signal import find_peaks


def write_ap_features():
    all_files = listdir('./data/cells')

    all_ap_features = []
    all_currs = []

    for f in all_files:
        if '.DS' in f:
            continue

        curr_ap_features = [f] + [f.split('_')[-1]] + get_ap_features(f)
        all_ap_features.append(curr_ap_features)

        
    all_ap_features = np.array(all_ap_features)
    all_ap_features = pd.DataFrame(all_ap_features, columns=['File', 'Drug', 'MP', 'APD90', 'CL', 'dVdt', 'Peak', 'Amplitude', 'APD20', 'MP_Drug', 'APD90_Drug', 'CL_Drug', 'dVdt_Drug', 'Peak_Drug', 'Amplitude_Drug', 'APD20_Drug'])

    all_ap_features.to_csv('./data/ap_features.csv', index=False)


def get_ap_features(f):
    #mp, apd90, apa, dvdt, peak, cl
    mp, apd90, cl, dvdt, peak, apa, apd20 = None, None, None, None, None, None, None
    mp_drug, apd90_drug, cl_drug, dvdt_drug, peak_drug, apa_drug, apd20_drug = None, None, None, None, None, None, None

    ap_dat = pd.read_csv(f'./data/cells/{f}/Pre-drug_spont.csv')
    apd90 = get_apd(ap_dat)
    apd20 = get_apd(ap_dat, 20)

    if apd90 is not None:
        cl = get_cl(ap_dat)
        dvdt = get_dvdt(ap_dat)
        peak = (ap_dat['Voltage (V)'].max())*1000 

    mp = ap_dat['Voltage (V)'].min()*1000

    ap_dat = pd.read_csv(f'./data/cells/{f}/Post-drug_spont.csv')
    apd90_drug = get_apd(ap_dat)
    apd20_drug = get_apd(ap_dat, 20)

    if apd90_drug is not None:
        cl_drug = get_cl(ap_dat)
        dvdt_drug = get_dvdt(ap_dat)
        peak_drug = (ap_dat['Voltage (V)'].max())*1000 

    mp_drug = ap_dat['Voltage (V)'].min()*1000

    #return {'mp': mp, 'apd90': apd90, 'cl': cl, 'dvdt': dvdt, 'peak': peak}
    return [mp, apd90, cl, dvdt, peak, apa, apd20, mp_drug, apd90_drug, cl_drug, dvdt_drug, peak_drug, apa_drug, apd20_drug]


def get_apd(ap_dat, apd_num=90):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    if ((v.max() - v.min()) < 20):
        return None
    if v.max() < 0:
        return None

    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    v_smooth = np.convolve(v, kernel, mode='same')

    peak_idxs = find_peaks(np.diff(v_smooth), height=.1, distance=1000)[0]

    if len(peak_idxs) < 2:
        return None

    min_v = np.min(v[peak_idxs[0]:peak_idxs[1]])
    min_idx = np.argmin(v[peak_idxs[0]:peak_idxs[1]])
    search_space = [peak_idxs[0], peak_idxs[0] + min_idx]
    amplitude = np.max(v[search_space[0]:search_space[1]]) - min_v
    v_90 = min_v + amplitude * (1-apd_num/100)
    idx_apd90 = np.argmin(np.abs(v[search_space[0]:search_space[1]] - v_90))

    return idx_apd90 / 10


def get_cl(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    peak_pts = find_peaks(v, 10, distance=1000, width=200)[0]

    average_cl = np.mean(np.diff(peak_pts)) / 10

    return average_cl


def get_dvdt(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    #plt.plot(t, v)

    peak_pts = find_peaks(v, 10, distance=1000, width=200)[0]

    #plt.plot(t, v)

    new_v = moving_average(v, 10)
    new_t = moving_average(t, 10)

    #plt.plot(np.diff(new_v))

    v_diff = np.diff(new_v)

    dvdt_maxs = []

    for peak_pt in peak_pts:
        start_pt = int(peak_pt/10-50)
        if start_pt < 0:
            continue
        end_pt = int(peak_pt/10)
        #try:
        dvdt_maxs.append(np.max(v_diff[start_pt:end_pt]))
        #except:
        #    import pdb
        #    pdb.set_trace()

        #plt.axvline(peak_pt/10, -50, 20, c='c')
        #plt.axvline(peak_pt/10-50, -50, 20, c='r')

    average_dvdt = np.mean(dvdt_maxs)

    return average_dvdt


def get_amplitude(ap_dat):
    t = ap_dat['Time (s)'].values * 1000
    v = ap_dat['Voltage (V)'].values * 1000

    peak_pts = find_peaks(v, 10, distance=1000, width=200)[0][1:5]
    min_pts = find_peaks(-v, 10, distance=1000, width=200)[0][1:4]

    if min_pts[0] > peak_pts[0]:
        peak_pts = peak_pts[1:]

    #points = np.array([t[min_pts[0]:peak_pts[0]], v[min_pts[0]:peak_pts[0]]])

    points = [[t[min_pts[0]+i], v[min_pts[0]+i]] for i in range(0, peak_pts[0]-min_pts[0])]

    best_idx = longest_perpendicular([t[min_pts[0]], v[min_pts[0]]],
                            [t[peak_pts[0]], v[peak_pts[0]]],
                            points)

    import matplotlib.pyplot as plt

    plt.plot(t, v)
    #[print(t[p]) for p in min_pts]
    #[plt.axvline(t[p], color='r') for p in min_pts]
    #[plt.axvline(t[p], color='g') for p in peak_pts]
    [plt.plot([t[p], t[peak_pts[i]]], [v[p], v[peak_pts[i]]]) for i, p in enumerate(min_pts)]

    plt.axvline(t[best_idx+min_pts[0]])
    plt.show()

    import pdb
    pdb.set_trace()

    new_v = moving_average(v, 10)
    new_t = moving_average(t, 10)

    v_diff = np.diff(new_v)

    dvdt_maxs = []

    for peak_pt in peak_pts:
        start_pt = int(peak_pt/10-50)
        end_pt = int(peak_pt/10)
        dvdt_maxs.append(np.max(v_diff[start_pt:end_pt]))

    average_dvdt = np.mean(dvdt_maxs)


    return average_dvdt


def moving_average(x, n=10):
    idxs = range(n, len(x), n)
    new_vals = [x[(i-n):i].mean() for i in idxs]
    return np.array(new_vals)


def longest_perpendicular(p1, p2, points):
    def perpendicular_distance(point, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x, y = point
        # calculate the denominator of the equation
        denom = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        # if the line is vertical, the denominator will be 0 and we cannot divide by it
        if denom == 0:
            return abs(x - x1)
        # calculate the numerator of the equation
        num = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        # calculate the distance using the equation
        distance = num / denom
        return distance

    # set the initial maximum distance and the index to None
    max_distance = None
    index = None
    # iterate through the points
    for i, point in enumerate(points):
        # calculate the perpendicular distance from the point to the line
        distance = perpendicular_distance(point, p1, p2)
        # if the distance is larger than the current maximum distance, update the maximum distance and the index
        if max_distance is None or distance > max_distance:
            max_distance = distance
            index = i
    # return the index of the point with the longest perpendicular line
    return index


def main():
    write_ap_features()
    

if __name__ == '__main__':
    main()

