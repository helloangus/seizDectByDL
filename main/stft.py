# 调整信息显示等级，越高信息越少
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='-1'  # 只使用第二块GPU

import time
start =time.process_time()

# 调用相关库
import tensorflow as tf
import numpy as np

# 调用数据
data_path = '/home/lichengchen/data/training_data/'

### 训练参数定义    ###
window_lenth = 30
sfreq = 256
num_of_channels = 23-2
iteration = 80
batch_size = 50
step_to_show_info = 20

save_path = 'save/2nd_GCN/'
init_adj_mat = np.loadtxt(open('adj_mat.csv',"rb"),delimiter=",",skiprows=0) 

# 生成数据随机矩阵
num_of_files = len(os.listdir(data_path)) - 1    # 获取目录下所有文件数
random_array = np.linspace(0, num_of_files-1, num_of_files)
random_array = random_array.reshape(num_of_files, 1)    # 随机化
temp = np.random.permutation(random_array)
random_array = np.append(random_array, temp, axis=1)    # 沿着列方向增加随机化的标号

# 定义获取数据函数
used_numName_of_file = 0    # 已经使用过了的数据数量
def get_data(num_of_batch):
    global used_numName_of_file
    global random_array
    for i in range(num_of_batch):
        numName_of_file = int(random_array[used_numName_of_file] [1])   # 从随机数组中得到本次抽取的数据
        data = np.load(data_path + 'chb_' + str(numName_of_file) +'.npz')  # 读取文件
        temp = data['data'].reshape(num_of_channels+2, window_lenth*sfreq)  # 获取x，并展开成矩阵
        temp = np.delete(temp, [18, 22], axis=0)                            #删除重复的19和23通道
        temp = temp.reshape(1, window_lenth*sfreq*num_of_channels)            #重新展开成行向量
        if i == 0:
            batch_xs = temp
        else:
            batch_xs = np.concatenate((batch_xs, temp), axis=0)

        if data['with_seiz'] == 1:  # 有癫痫记录为[[1, 0]], 无癫痫记录为[[0, 1]]
            if i == 0:
                batch_ys = np.array([[0, 1]])
            else:
                batch_ys = np.concatenate((batch_ys, np.array([[0, 1]])), axis=0)
        else:
            if i == 0:
                batch_ys = np.array([[1, 0]])
            else:
                batch_ys = np.concatenate((batch_ys, np.array([[1, 0]])), axis=0)
        used_numName_of_file = used_numName_of_file + 1
    return batch_xs, batch_ys

config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # 按需调节显存占用
# config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 强制显存占用为40%
sess = tf.Session()

batch_xs, batch_ys = get_data(batch_size)

result = tf.contrib.signal.stft(tf.cast(tf.reshape(batch_xs, [-1, 21, 30*256]), tf.float32), frame_length=300, frame_step=58)

print(result)






