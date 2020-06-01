import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.io import concatenate_raws, read_raw_edf

# 相关参数
window_lenth = 30
sfreq = 256

save_path = '/Users/anguslee/chb/resample/test'
with_seiz = 1



load_path = '/Users/anguslee/chb/24/chb24_21.edf'
start_time = 2900


raw = read_raw_edf(load_path, preload=True)
print(raw.ch_names)
picks = mne.pick_types(raw.info, eeg=True, exclude='bads')


t_idx = raw.time_as_index([start_time, start_time+window_lenth])
data,times=raw[picks, t_idx[0]:t_idx[1]]

np.savez_compressed(save_path+'test', data=data, with_seiz = with_seiz ,times=times)


npzfile=np.load(save_path+ 'test' +'.npz')

data = npzfile['data']
with_seiz = npzfile['with_seiz']
times = npzfile['times']

plt.plot(times,data.T)
plt.title("Sample channels")
plt.show()