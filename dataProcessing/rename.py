import os
import os.path
import shutil
rootdir = '/Users/anguslee/chb/chb03/'                                   # 指明被遍历的文件夹
desdir = '/Users/anguslee/full/'

i = 9096
filenames = os.listdir(rootdir)
for filename in filenames:
    if filename.endswith('.npz'):
        shutil.copy(os.path.join(rootdir, filename), os.path.join(desdir, 'chb_' + str(i) + '.npz'))
        i = i + 1
        
print(i)