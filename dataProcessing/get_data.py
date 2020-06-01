import mne
import numpy as np
import matplotlib.pyplot as plt
from mne.io import concatenate_raws, read_raw_edf

# 相关参数
window_lenth = 30
sfreq = 256

save_path = '/Users/anguslee/chb/resample/rechb_'
with_seiz = 1



load_path = '/Users/anguslee/chb/24/chb24_21.edf'
start_name = 5522
start_start_time = 2780
end_time = 2900

# startnpz=np.load('/Users/anguslee/chb/start_para' +'.npz')
# start_name = startnpz['start_name']
# start_time = startnpz['start_time']

# start_name = 150
# start_time = 1440

# print('start_name is ', start_name, '\n')
# print('start_time is ', start_time, '\n')

start_time = start_start_time
while start_time < start_start_time+30:


    raw = read_raw_edf(load_path, preload=True)
    picks = mne.pick_types(raw.info, eeg=True, exclude='bads')

    time = start_time
    while time+30 <end_time :
        t_idx = raw.time_as_index([time, time+window_lenth])
        data,times=raw[picks, t_idx[0]:t_idx[1]]
        if times.shape == (sfreq*window_lenth,):
            np.savez_compressed(save_path+str(start_name), data=data, with_seiz = with_seiz ,times=times)
            start_name = start_name + 1
        else:
            break
        time = time + 30
        

    npzfile=np.load(save_path+ str(start_name-1) +'.npz')

    data = npzfile['data']
    with_seiz = npzfile['with_seiz']
    times = npzfile['times']


    print(data.shape)
    print(with_seiz)
    start_time = start_time + 1
    print('\nstart_name will be changed to ', start_name)

    

print('loop ended')

print('\nstart_name will be changed to ', start_name)
# print('\nstart_time will be changed to ', start_time+1)

# np.savez_compressed('/Users/anguslee/chb/start_para', start_name=start_name, start_time=start_time+1)


# load_data = npzfile['data']
# print(load_data[:,:])

# plt.plot(times,data.T)
# plt.title("Sample channels")
# plt.show()