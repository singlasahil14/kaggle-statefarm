from utils import *

orig_path = 'data/original/'
path = 'data/'

# Create dirs
from shutil import rmtree
os.chdir(orig_path+'train/')
for d in glob('c?'):
    train_dir = '../../train/'+d
    valid_dir = '../../valid/'+d
    if os.path.exists(train_dir):
        rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.exists(valid_dir):
        rmtree(valid_dir)
    os.makedirs(valid_dir)
os.chdir('../../../')

# Groupwise split
drivers = pd.read_csv(orig_path+'driver_imgs_list.csv')
from sklearn.model_selection import LeavePGroupsOut
gkf = LeavePGroupsOut(n_groups=3)
X = drivers['img'].values
y = drivers['classname'].values
groups = drivers['subject'].values
train_indices, val_indices = next(gkf.split(X, groups=groups))

#For debugging
train_list = zip(X[train_indices].tolist(),y[train_indices].tolist(),groups[train_indices].tolist())
val_list = zip(X[val_indices].tolist(),y[val_indices].tolist(),groups[val_indices].tolist())
f=open('train_list.txt','w')
for ele in train_list:
  f.write(str(ele)+'\n')
f.close()
f=open('val_list.txt','w')
for ele in val_list:
   f.write(str(ele)+'\n')
f.close()

# Create filenames
train_files = map('/'.join, zip(y[train_indices], X[train_indices]))
val_files = map('/'.join, zip(y[val_indices], X[val_indices]))

# Copy files
from shutil import copyfile
for file_name in train_files: 
    copyfile(orig_path + 'train/' + file_name, path + 'train/' + file_name)
for file_name in val_files:
    copyfile(orig_path + 'train/' + file_name, path + 'valid/' + file_name)
