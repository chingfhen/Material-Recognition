import numpy as np
import os
from numpy import asarray
from PIL import Image

"""
Returns X_train, y_train, X_test, y_test 
    - shuffled numpy arrays
    
Required Inputs:
    1. data_dir 
        - directory containing the folders "fabric", "foliage", "glass" etc...
    2. new_size 
        - new size of image e.g (100,100)
"""
def load_FMD(data_dir, new_size, train_size = 0.8, seed = 0):
    
    X = []
    y = []
    label_names = ["fabric", "foliage", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
    label_encoding  = dict(zip(label_names,range(len(label_names))))
    
    # loop through the folders
    for label in label_names:
        for filename in os.listdir(os.path.join(data_dir,f"{label}")):
            # read image and its label as numpy array  
            image_data = read_image(os.path.join(data_dir,f"{label}\\{filename}"), new_size = new_size)
            if image_data is not None:
                X.append(image_data)
                y.append(label_encoding[label])
            
           
        
    # shuffle and split into train and test sets
    X,y = np.array(X),np.array(y)        
    np.random.seed(seed)
    indices = np.array(range(0,len(y)))
    np.random.shuffle(indices)
    split_index = int(train_size*len(y))
    X_train, y_train, X_test, y_test = X[indices][:split_index],y[indices][:split_index],X[indices][split_index:],y[indices][split_index:]
    
    return X_train, y_train, X_test, y_test

"""
WRAPPED INSIDE load_FMD
returns image as array
returns None if image is invalid
    - black and white images are invalid
"""
def read_image(path, new_size = None, to_array = True):
    if "jpg" not in path:
        return
    try:
        if new_size is not None:
            image = Image.open(path).resize(new_size)
        else:
            image = Image.open(path)
    except:
        return
    if to_array:
        arr = asarray(image)
        if len(arr.shape)==2:
            return
        return arr
    else:
        return image