# COMP9517: Computer Vision 2024 T3 Project Method: UDTransNet

#### z5499630 Hudson(Boyang) Peng

## We recommend running this method on Google Colab

- The pretrained three model weights using UDTransNet an be accessed by using this [Google Drive link](https://drive.google.com/file/d/1QuTORyzaZLMyP7VNFUPkaj-IbM9P3rur/view?usp=drive_link)

## Running Guide:

### Step 1: Clone the UDTransNet model from github

```bash
git clone https://github.com/McGregorWwww/UDTransNet.git
```

### Step 2: Install the required packages

```bash
pip install -r requirements.txt

pip install tensorboardX
pip install ml_collections
```


### Step 3: Download the SeaTurtleID2022 dataset from kaggle

```bash
curl -L -o ./archive.zip https://www.kaggle.com/api/v1/datasets/download/wildlifedatasets/seaturtleid2022
```


### Step 4: Generate masks of the dataset with areas in ['turtle', 'head', 'flippers']

- In this method, I need to partition the turtle into three distinct areas to perform binary segmentation over each area

### Step 5: Divide all images into a training folder and a testing folder based on the 'split_closed' column in the CSV file

- See Codes in `9517_UDTransNet_pby.ipynb` Step 5

### Step 6: Create training and testing folders for the models

- The `“img”` folder in both training and testing folders is intended to store original images.
- The `“labelcol”` folder in both training and testing folders is intended to store masked ground truth images.

Then prepare the datasets in the following format for easy use of the code:

- For Testing, change the name `<area>_test_folder` to `img`, and change the name `<area>_test_folder_mask` to `labelcol`.

- For Training, change the name `<area>_train_valid_folder` to `img`, and change the name `<area>_train_valid_folder_mask` to `labelcol`.

```text
├── datasets
│   ├── Turtle
│   │   ├── Test_Folder
│   │   │   ├── img
│   │   │   └── labelcol
│   │   └── Train_Folder
│   │       ├── img
│   │       └── labelcol
│   ├── Turtle_H
│   │   ├── Test_Folder
│   │   │   ├── img
│   │   │   └── labelcol
│   │   └── Train_Folder
│   │       ├── img
│   │       └── labelcol
│   └── Turtle_F
│       ├── Test_Folder
│       │   ├── img
│       │   └── labelcol
│       └── Train_Folder
│           ├── img
│           └── labelcol
```


### Move the split images into the corresponding folders as specified above


### Step 7: Start the training

The first step is to change the settings in `Config.py`, all the configurations including learning rate, batch size and etc. are in it.

```python
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
```

We optimize the convolution parameters in U-Net and the DAT parameters together with a single loss. Run:

```bash
python train_kfold.py
```

The results including log files, model weights, etc., are in '[TaskName]_kfold' folder, e.g., 'Turtle_kfold'.


### Step 8: Start Testing

First, change the session name in `Config.py` as the training phase.

```python
# Choose the model
model_name = 'UDTransNet'

# Change the test_session name into the one you trained
# Or you can use the one we trained for the project
if task_name == "Turtle":
    if model_name == "UDTransNet":
        test_session = "Test_session_[your_datetime]"

elif task_name == "Turtle_H":
    if model_name == "UDTransNet":
        test_session = "Test_session_[your_datetime]"

elif task_name == "Turtle_F":
    if model_name == "UDTransNet":
        test_session = "Test_session_[your_datetime]"
```

Then, for Turtle, Turtle_H and Turtle_F, run:

```bash
python test_kfold.py
```

### You can get the Dice and IoU scores and the visualization results `in the visualized folder`

### Here is the table containing all the results from three parts
| Metrics   | Head  | Carapace | Flippers | 
|-|-|-|-|
|**DICE**|`88.50%`|`97.41%`|`89.47%`|
|**mIoU**|`81.48%`|`95.31%`|`82.43%`|