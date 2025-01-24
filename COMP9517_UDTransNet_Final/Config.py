import os
import torch
import time

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)
kfold = 5
cosineLR = True
n_channels = 3
n_labels = 1
epochs = 500
img_size = 224
img_size2 = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 5000

# Choose your segmentation task
task_name = 'Turtle'
# task_name = 'Turtle_H'
# task_name = 'Turtle_F'


if task_name == "Turtle":
    learning_rate = 1e-3
    early_stopping_patience = 15
    batch_size = 24
    n_labels = 1
    n_channels = 3
elif task_name == "Turtle_H":
    learning_rate = 1e-3
    early_stopping_patience = 10
    batch_size = 32
    n_labels = 1
    n_channels = 3
elif task_name == "Turtle_F":
    learning_rate = 1e-3
    early_stopping_patience = 10
    batch_size = 32
    n_labels = 1
    n_channels = 3

# Choose the model
model_name = 'UDTransNet'
# model_name = 'UNet'

# Change the test_session name into the one you trained
# Or you can use the one we trained for the project
if task_name == "Turtle":
    if model_name == "UDTransNet":
        test_session = "Test_session_10.25_00h25"
    if model_name == "UNet":
        test_session = "Test_session_10.27_02h23"

elif task_name == "Turtle_H":
    if model_name == "UDTransNet":
        test_session = "Test_session_10.25_05h19"
    if model_name == "UNet":
        test_session = "Test_session_10.27_03h17"

elif task_name == "Turtle_F":
    if model_name == "UDTransNet":
        test_session = "Test_session_10.26_10h23"
    if model_name == "UNet":
        test_session = "Test_session_10.27_04h15"


# Dataset Paths Configs
train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'

# Parameters Configs
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'_kfold/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'