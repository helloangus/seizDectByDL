# 在终端中输入 tensorboard --logdir=logs    #

### import part ###
# 调整信息显示等级，越高信息越少
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='2'  # 只使用第二块GPU

import time
start = time.process_time()

# 调用相关库
import tensorflow as tf
import numpy as np
import shutil

# 调用数据
data_path = '/home/lichengchen/data/training_data/'

init_adj_mat = np.loadtxt(open('adj_mat.csv',"rb"),delimiter=",",skiprows=0) 

print('\n\n\nImport complete!\n')

### import part ###



### 训练参数定义    ###
window_lenth = 30
sfreq = 256
num_of_channels = 23 - 2
input_size = sfreq * window_lenth * num_of_channels
output_size = 2
init_learningRate = 0.01 * 2
iteration = 700
batch_size = 100
step_to_change_learningRate = 100
step_to_show_info = 50
step_to_record_info = 10

test_iteration = 0

save_path = 'save/2nd_GCN/'
epsilon = 1e-3
decay = 0.9

### 训练参数定义    ###



### def part    ###
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


# 定义修改学习率函数
def change_learningRate(iteration, learningRate_op):  
    # 使用tf.cast((iteration%500 < 1), tf.float32)来判断已经达到一定的iteration（==不可用）
    learningRate_op = learningRate_op / (tf.cast((iteration%step_to_change_learningRate < 1), tf.float32) + 1)  
    return learningRate_op

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # padding 'SAME'与原图片大小一致（包含零填充）  'VALID'比原图小
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.layers.max_pooling2d(x, pool_size=[2,2], strides=[2,2], padding='SAME')

# 获取归一化参数
def get_moments(inputs, is_training, axes):
    fc_mean, fc_var = tf.nn.moments(inputs, axes=axes)
    ema = tf.train.ExponentialMovingAverage(decay=decay)  # exponential moving average 的 decay 度
    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = tf.cond(is_training,    # on_train 的值是 True/False
                        mean_var_with_update,   # 如果是 True, 更新 mean/var
                        lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                            ema.average(fc_mean), 
                            ema.average(fc_var)
                            )    
                        )
    return mean, var

# 批归一化层
def bn_layer(inputs, mean, var, input_size, output_size):
    scale = tf.Variable(tf.ones(input_size))
    shift = tf.Variable(tf.zeros(input_size))
    return tf.reshape(tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon), output_size)

# 聚合操作（空域卷积）层
def ag_layer(inputs, none_ch_size, output_size, adj_mat):
    local_adj_mat = adj_mat + 1 - 1     # 目的是生成一个新矩阵
    main_mom = tf.Variable(tf.constant(0.5))
    sub_mom = tf.Variable(tf.constant(0.5))
    for main_ch in range(num_of_channels):
        neb = tf.Variable(tf.zeros(none_ch_size))
        for sub_ch in range(num_of_channels):
            if(local_adj_mat[main_ch, sub_ch] != 0):       # 不是同一个通道或者已经聚合的通道进行操作
                cont = 0
                if local_adj_mat[main_ch, sub_ch] == 1 :
                    if cont == 0:
                        neb = inputs[:, :, :, sub_ch]
                    else:
                        neb = neb + inputs[:, :, :, sub_ch]   # 把所有的邻居加起来
                    cont = cont + 1
                    local_adj_mat[main_ch, sub_ch] = local_adj_mat[main_ch, sub_ch] - 1 # 改变邻接矩阵的值
                else:
                    local_adj_mat[main_ch, sub_ch] = local_adj_mat[main_ch, sub_ch] - 1

        temp = tf.reshape(tf.reshape(inputs[:, :, :, main_ch], none_ch_size) *
             main_mom + tf.reshape(neb, none_ch_size) * sub_mom, none_ch_size)
        
        if main_ch == 0:
            outputs = temp
        else:
            outputs = tf.concat([outputs, temp], 3)
            
    return tf.reshape(outputs, output_size), local_adj_mat, main_mom, sub_mom



### def part    ###



### placehouder part    ###
# 定义输入placeholder，并在tensorboard中成组为input
xs = tf.placeholder(tf.float32, [None, input_size], name='xs')
ys = tf.placeholder(tf.float32, [None, output_size], name='ys')

# 学习率相关placholder
it = tf.placeholder(tf.float32, name='iteration')
lr = tf.placeholder(tf.float32, name='learningRate_op')

is_training = tf.placeholder(tf.bool, name='is_training')

# 定义dropout中的保留比例
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

### placehouder part    ###



### generate network    ###

# 特征提取  （短时傅里叶变换）
result = tf.contrib.signal.stft(tf.cast(tf.reshape(xs, [-1, 21, 30*256]), tf.float32), 
frame_length=300, frame_step=58, fft_length=255)
x_image2 = tf.reshape(tf.transpose(tf.reshape(result, 
    [batch_size, num_of_channels, -1]), [0, 2, 1]), [batch_size, 128, 128, num_of_channels])   # [num_of_samples, x, y, z]
x_image1 = tf.math.abs(x_image2)
x_image = x_image1[:, :, :, :]

# 输入数据归一化处理
mom_xs = get_moments(x_image, is_training=is_training, axes=[0, 1, 2])
xs_nor = bn_layer(x_image, mom_xs[0], mom_xs[1], [128, 128, num_of_channels], [batch_size, 128, 128, num_of_channels])

# conv1 layer
W_conv1 = weight_variable([5, 5, 21, 21])                    # [x, y, z, out_z]
b_conv1 = bias_variable([21])
fdc_c1 = conv2d(xs_nor, W_conv1) + b_conv1
adc_c1 = ag_layer(fdc_c1, [batch_size, 128, 128, 1], [batch_size, 128, 128, 21], init_adj_mat)
mom_c1 = get_moments(adc_c1[0], is_training=is_training, axes=[0, 1, 2, 3])
bn_c1 = bn_layer(adc_c1[0], mom_c1[0], mom_c1[1], [128, 128, 21], [batch_size, 128, 128, 21])       # 128*128*21
h_conv1 = tf.nn.relu(bn_c1)
h_pool1 = max_pool_2x2(h_conv1)                             # output size 64*64*21

# conv2 layer
W_conv2 = weight_variable([5, 5, 21, 21])                   # [x, y, z, out_z]
b_conv2 = bias_variable([21])
fdc_c2 = conv2d(h_pool1, W_conv2) + b_conv2
adc_c2 = ag_layer(fdc_c2, [batch_size, 64, 64, 1], [batch_size, 64, 64, 21], adc_c1[1])
mom_c2 = get_moments(adc_c2[0], is_training=is_training, axes=[0, 1, 2, 3])
bn_c2 = bn_layer(adc_c2[0], mom_c2[0], mom_c2[1], [64, 64, 21], [batch_size, 64, 64, 21])
h_conv2 = tf.nn.relu(bn_c2)    
h_pool2 = max_pool_2x2(h_conv2)                             # output size 32*32*21

# conv3 layer
W_conv3 = weight_variable([5, 5, 21, 21])                   # [x, y, z, out_z]
b_conv3 = bias_variable([21])
fdc_c3 = conv2d(h_pool2, W_conv3) + b_conv3
adc_c3 = ag_layer(fdc_c3, [batch_size, 32, 32, 1], [batch_size, 32, 32, 21], adc_c2[1])
mom_c3 = get_moments(adc_c3[0], is_training=is_training, axes=[0, 1, 2, 3])
bn_c3 = bn_layer(adc_c3[0], mom_c3[0], mom_c3[1], [32, 32, 21], [batch_size, 32, 32, 21])
h_conv3 = tf.nn.relu(bn_c3)    
h_pool3 = max_pool_2x2(h_conv3)                             # output size 16*16*21

# conv4 layer
W_conv4 = weight_variable([5, 5, 21, 21])                   # [x, y, z, out_z]
b_conv4 = bias_variable([21])
fdc_c4 = conv2d(h_pool3, W_conv4) + b_conv4
adc_c4 = ag_layer(fdc_c4, [batch_size, 16, 16, 1], [batch_size, 16, 16, 21], adc_c3[1])
mom_c4 = get_moments(adc_c4[0], is_training=is_training, axes=[0, 1, 2, 3])
bn_c4 = bn_layer(adc_c4[0], mom_c4[0], mom_c4[1], [16, 16, 21], [batch_size, 16, 16, 21])
h_conv4 = tf.nn.relu(bn_c4)    
h_pool4 = max_pool_2x2(h_conv4)                             # output size 8*8*21

# func1 layer
W_fc1 = weight_variable([8*8*21, 1024])
b_fc1 = bias_variable([1024])
conv_flat = tf.reshape(h_pool4, [batch_size, 8*8*21])              # [-1, 8, 8, 21] ->> [-1, 8*8*21]
fc1 = tf.matmul(conv_flat, W_fc1) + b_fc1
mom_f1 = get_moments(fc1, is_training=is_training, axes=[0])
bn_f1 = bn_layer(fc1, mom_f1[0], mom_f1[1], [1024], [batch_size, 1024])
h_fc1 = tf.nn.relu(bn_f1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# func2 layer
W_fc2 = weight_variable([1024, 512])
b_fc2 = bias_variable([512])
fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
mom_f2 = get_moments(fc2, is_training=is_training, axes=[0])
bn_f2 = bn_layer(fc2, mom_f2[0], mom_f2[1], [512], [batch_size, 512])
h_fc2 = tf.nn.relu(bn_f2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

# func3 layer
W_fc3 = weight_variable([512, output_size])
b_fc3 = bias_variable([output_size])
fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
mom_f3 = get_moments(fc3, is_training=is_training, axes=[0])
bn_f3 = bn_layer(fc3, mom_f3[0], mom_f3[1], [output_size], [batch_size, output_size])
prediction = tf.nn.softmax(bn_f3)

### generate network    ###



### define parameters   ###
# 定义学习率
with tf.name_scope('learning_rate'): 
    learningRate = tf.Variable(init_learningRate, name='learningRate')  # 用于更新学习率是暂存
    tf.summary.scalar('learning_rate', learningRate)
# 定义更新学习率的操作
learningRate_op = change_learningRate(it, lr)

# 定义cross_entropy和优化器、学习率
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
        (labels=ys, logits=prediction, name='cross_entropy'))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learningRate_op).minimize(cross_entropy)

print('Define complete!\n')

# 创建saver
saver = tf.train.Saver()
# 假如需要保存y，以便在预测时使用
tf.add_to_collection('pred_network', prediction)

### define parameters   ###



### initializer ###
# 初始化所有变量
init = tf.global_variables_initializer()
# 初始化对话
config = tf.ConfigProto()
# config.gpu_options.allow_growth = True  # 按需调节显存占用
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 强制显存占用为90%
sess = tf.Session(config=config)
sess.run(init)
# 合并并写入所有summary
merged = tf.summary.merge_all()   
path = 'logs'
shutil.rmtree(path) # 递归删除该目录下所有文件夹和文件
train_writer = tf.summary.FileWriter(path+'/train/2/', sess.graph)    # 输出tensorboard信息
# test_writer = tf.summary.FileWriter(path+'/test', sess.graph)

print('Init complete!\n')

### initializer ###



### training step   ###
# 进行训练
print('\n\n\nTraining start!\n')
num_of_TN = 0
num_of_FP = 0
num_of_FN = 0
num_of_TP = 0

for i in range(iteration):
    batch_xs, batch_ys = get_data(batch_size)

    # 进行每步训练
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, it: i, 
        lr: sess.run(learningRate), keep_prob: 0.7, is_training: True})
  

    ### 计算评价指标    ###
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

    ### 计算评价指标    ###



    # 更新学习率
    new_value = sess.run(learningRate_op, feed_dict={it: i, lr: sess.run(learningRate)})
    update = tf.assign(learningRate, new_value)
    sess.run(update)

    # 打印相关信息
    if (i+1) % step_to_show_info == 0:
        print('Iteration:     ', i+1)
        print('Number of data per iteration: ', batch_size)
        print('Learning_rate: ', sess.run(learningRate))
        print('Cross_entropy: ', sess.run(cross_entropy, 
            feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1, is_training: False}))
        
        TPR = num_of_TP / (num_of_TP + num_of_FN)   # 灵敏度，TPR = TP/(TP+FN)
        TNR = num_of_TN / (num_of_TN + num_of_FP)   # 特异性，TNR = TN/(TN+FP)
        FPR = tf.cast(num_of_FP, tf.float32) / (i * batch_size / (60/window_lenth*60))   # 误报率， FPR = FP/Total time(/h)
        
        print('TPR: ', sess.run(TPR))
        print('TNR: ', sess.run(TNR))
        print('FPR: ', sess.run(FPR), '\n')
        print('num_of_TN: ', sess.run(num_of_TN))
        print('num_of_FP: ', sess.run(num_of_FP))
        print('num_of_FN: ', sess.run(num_of_FN))
        print('num_of_TP: ', sess.run(num_of_TP), '\n')

        nowTime = time.process_time()
        print('Running time:   %s Seconds'%(nowTime-start), '\n\n\n')

        saver.save(sess, save_path+'2nd_GCN-model', global_step=i+1)  
        num_of_TN = 0
        num_of_FP = 0
        num_of_FN = 0
        num_of_TP = 0

        
    # 记录summary
    if (i+1) % step_to_record_info == 0:
        train_result = sess.run(merged, 
            feed_dict={xs: batch_xs, ys: batch_ys, it: i, lr: sess.run(learningRate), keep_prob: 1, is_training: False})
        train_writer.add_summary(train_result, i)
    
print('Trainning finish!\n')
end = time.process_time()
print('Total time: %s Seconds\n\n\n'%(end-start))
### training step   ###



### testing step    ###
print('Testing ... ...\n')

test_num_of_TN = 0
test_num_of_FP = 0
test_num_of_FN = 0
test_num_of_TP = 0
for i in range(test_iteration):
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

    test_num_of_TN = test_num_of_TN + tf.reduce_sum(TN_or_no)
    test_num_of_FP = test_num_of_FP + tf.reduce_sum(FP_or_no)
    test_num_of_FN = test_num_of_FN + tf.reduce_sum(FN_or_no)
    test_num_of_TP = test_num_of_TP + tf.reduce_sum(TP_or_no)

    if (i+1) % step_to_show_info == 0:
        TPR = test_num_of_TP / (test_num_of_TP + test_num_of_FN)   # 灵敏度，TPR = TP/(TP+FN)
        TNR = test_num_of_TN / (test_num_of_TN + test_num_of_FP)   # 特异性，TNR = TN/(TN+FP)
        FPR = tf.cast(test_num_of_FP, tf.float32) / (i * batch_size / (60/window_lenth*60))   # 误报率， FPR = FP/Total time(/h)
        print('Iteration: ', i+1)
        print('Number of data per iteration: ', batch_size, '\n')
        print('TPR: ', sess.run(TPR))
        print('TNR: ', sess.run(TNR))
        print('FPR: ', sess.run(FPR))
        print('\n')
        print('test_num_of_TN: ', sess.run(test_num_of_TN))
        print('test_num_of_FP: ', sess.run(test_num_of_FP))
        print('test_num_of_FN: ', sess.run(test_num_of_FN))
        print('test_num_of_TN: ', sess.run(test_num_of_TP), '\n')

        test_nowTime = time.process_time()
        print('Running time: %s Seconds'%(test_nowTime-end), '\n')


print('TPR: ', sess.run(TPR))
print('TNR: ', sess.run(TNR))
print('FPR: ', sess.run(FPR))
print('\n')
print('test_num_of_TN: ', sess.run(test_num_of_TN))
print('test_num_of_FP: ', sess.run(test_num_of_FP))
print('test_num_of_FN: ', sess.run(test_num_of_FN))
print('test_num_of_TN: ', sess.run(test_num_of_TP), '\n')

test_end = time.process_time()
print('Test time: %s Seconds\n\n\n'%(test_end-end))

### testing step    ###

