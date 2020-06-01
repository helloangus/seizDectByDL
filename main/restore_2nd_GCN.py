# 调整信息显示等级，越高信息越少
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='1'  # 只使用第二块GPU

import time
start =time.process_time()

# 调用相关库
import tensorflow as tf
import numpy as np

# 调用数据
data_path = '/home/lichengchen/data/test_data/'

### 训练参数定义    ###
window_lenth = 30
sfreq = 256
num_of_channels = 23-2
iteration = 50
batch_size = 100
step_to_show_info = 10

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
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 强制显存占用为40%
sess = tf.Session()

# 读取存储的模型
model = tf.train.import_meta_graph(save_path+'2nd_GCN-model-700.meta')
model.restore(sess, save_path+'2nd_GCN-model-700')
prediction = tf.get_collection('pred_network')[0]
graph = tf.get_default_graph()

# 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，而这些需要通过graph的get_operation_by_name方法来获取。
keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]
xs = graph.get_operation_by_name('xs').outputs[0]
is_training = graph.get_operation_by_name('is_training').outputs[0]


# 预测
print('\n')
num_of_TN = 0
num_of_FP = 0
num_of_FN = 0
num_of_TP = 0
for i in range(iteration):
    batch_xs, batch_ys = get_data(batch_size)
    result_pre = tf.argmax(sess.run(prediction, feed_dict={xs:batch_xs, keep_prob:1.0, is_training: False}), 1)
    result_raw = tf.argmax(batch_ys, 1)

    # 真实值为0，矩阵对应值为True
    condition_rawFalse = tf.equal(result_raw, 0)
    # 真实值为1，矩阵对应值为True
    condition_rawTrue = tf.equal(result_raw, 1)

    # 预测值为0时，矩阵对应值为1，求TN、FN用
    preFalse = tf.cast(tf.equal(result_pre, 0), tf.int32)
    # 预测值为1时，矩阵对应值为1，求TP、FP用
    preTrue = tf.cast(tf.equal(result_pre, 1), tf.int32)
    # 生成全0矩阵
    pre_zero = tf.zeros_like(preFalse)

    # 求TN,raw=0, pre=0
    TN_or_no = tf.where(condition_rawFalse, preFalse, pre_zero)
    # 求FP,raw=0, pre=1
    FP_or_no = tf.where(condition_rawFalse, preTrue, pre_zero)
    # 求FN,raw=1, pre=0
    FN_or_no = tf.where(condition_rawTrue, preFalse, pre_zero)
    # 求TP,raw=1, pre=1
    TP_or_no = tf.where(condition_rawTrue, preTrue, pre_zero)

    num_of_TN = num_of_TN + tf.reduce_sum(TN_or_no)
    num_of_FP = num_of_FP + tf.reduce_sum(FP_or_no)
    num_of_FN = num_of_FN + tf.reduce_sum(FN_or_no)
    num_of_TP = num_of_TP + tf.reduce_sum(TP_or_no)

    if (i+1) % step_to_show_info == 0:
        TPR = num_of_TP / (num_of_TP + num_of_FN)   # 灵敏度，TPR = TP/(TP+FN)
        TNR = num_of_TN / (num_of_TN + num_of_FP)   # 特异性，TNR = TN/(TN+FP)
        FPR = tf.cast(num_of_FP, tf.float32) / (i * batch_size / (60/window_lenth*60))   # 误报率， FPR = FP/Total time(/h)
        print('Iteration: ', i+1)
        print('Number of data per iteration: ', batch_size, '\n')
        print('TPR: ', sess.run(TPR))
        print('TNR: ', sess.run(TNR))
        print('FPR: ', sess.run(FPR))
        print('\n')
        print('num_of_TN: ', sess.run(num_of_TN))
        print('num_of_FP: ', sess.run(num_of_FP))
        print('num_of_FN: ', sess.run(num_of_FN))
        print('num_of_TP: ', sess.run(num_of_TP), '\n')

        nowTime = time.process_time()
        print('Running time: %s Seconds'%(nowTime-start), '\n\n\n')

end = time.process_time()
print('Total time: %s Seconds'%(end-start))
