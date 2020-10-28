# -----------------------------------------------------------------------------------------------------------
# ============================================ REQUIRED IMPORTS =============================================
# -----------------------------------------------------------------------------------------------------------

# ====== Regular Imports ======
import os, time
import numpy as np
import scipy as sp
import pandas as pd
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# # ======= Visualization ========
# import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import plotly.figure_factory as ff
# from plotly.subplots import make_subplots
# from plotly.offline import init_notebook_mode
# init_notebook_mode(connected=True)

# # ======= Images ========
# from VisualizerClass import Visualizer
import cv2
# from glob import glob
# from PIL import Image
# import albumentations as alb

# # ========= SKLearn =========
# from skimage.transform import resize
# from sklearn.model_selection import train_test_split

# # ======== Transfer Learnin ======
# from efficientnet.tfkeras import *
# from tensorflow.keras.applications import ResNet50, Xception

# # ======== TensorFlow ========
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import *
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.optimizers import *
# from tensorflow.keras.activations import *
# from tensorflow.python.keras.layers.advanced_activations import *
# from tensorflow.keras.initializers import *
# from tensorflow.keras.regularizers import *
# from tensorflow.keras.constraints import *
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.losses import *
# from tensorflow.keras.metrics import *
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

print(f"\n{'-' * 60} \n>>> ALL LIBRARIES SUCCESSFULLY IMPORTED \n{'-' * 60} \n")

# ===========================================================================================================
# ===========================================================================================================
# References
#
#
#
#
#
# ===========================================================================================================
# ===========================================================================================================


# -----------------------------------------------------------------------------------------------------------
# ========================================== SETTING UP BASE PATHs ==========================================
# -----------------------------------------------------------------------------------------------------------

base_path = "C:/Users/harsh/Desktop/Python/Projects/Pneumonia-Detection/"
base_path2 = "D:/Datasets/Brain Cancer Detection"

train_dir = os.path.join(base_path, "dataset/train/")
test_dir = os.path.join(base_path, "dataset/test/")
val_dir = os.path.join(base_path, "dataset/val/")
dataset_subdirs = [train_dir, test_dir, val_dir]

# -----------------------------------------------------------------------------------------------------------
# ========================================== ANALYZING IMAGE COUNTS =========================================
# -----------------------------------------------------------------------------------------------------------
def show_image_counts(dataset_subdirs):
    for dir in dataset_subdirs:
        normal_path = os.path.join(dir, "NORMAL")
        pneumonia_path = os.path.join(dir, "PNEUMONIA")

        normal_count = len(os.listdir(normal_path))
        pneumonia_count = len(os.listdir(pneumonia_path))

        print(f"{'-' * 85}")
        print(f">>> Counts in '{dir.split(sep='/')[-1].upper()}' SET:")
        print(f"Normal Images: {normal_count}")
        print(f"Pneumonia Images: {pneumonia_count}")         
        print(f"{'-' * 85}")


# -----------------------------------------------------------------------------------------------------------
# ============================================= IMAGE ANALYSIS ==============================================
# -----------------------------------------------------------------------------------------------------------

def extract_image_paths(dataset_subdirs):

    train_normal, train_pneumonia = [], []
    test_normal, test_pneumonia = [], []
    val_normal, val_pneumonia = [], []

    for i, dir in enumerate(dataset_subdirs):
        normal_path = os.path.join(dir, "NORMAL/")
        pneumonia_path = os.path.join(dir, "PNEUMONIA/")

        if i == 0:
            for path in os.listdir(normal_path):
                train_normal.append(os.path.join(normal_path, path))
            for path in os.listdir(pneumonia_path):
                train_pneumonia.append(os.path.join(pneumonia_path, path))

        elif i == 1:
            for path in os.listdir(normal_path):
                test_normal.append(os.path.join(normal_path, path))
            for path in os.listdir(pneumonia_path):
                test_pneumonia.append(os.path.join(pneumonia_path, path))

        elif i == 2:
            for path in os.listdir(normal_path):
                val_normal.append(os.path.join(normal_path, path))
            for path in os.listdir(pneumonia_path):
                val_pneumonia.append(os.path.join(pneumonia_path, path))
    
    train_paths = [train_normal, train_pneumonia]
    test_paths = [test_normal, test_pneumonia]
    val_paths = [val_normal, val_pneumonia]

    return [train_paths, test_paths, val_paths]

def get_image_resolution(image, format):
    img = cv2.imread(image)

    if format == "cv2_shape":    
        return img.shape
    
    elif format == "list":
        height, width, clr_channel = img.shape        
        return [height, width, clr_channel]

def scan_image_sizes(dataset_subdirs):
    paths = extract_image_paths(dataset_subdirs)
    smallest_shape, largest_shape = (5000, 5000, 3), (0, 0, 0)
    smallest_shape_path, largest_shape_path = "", ""

    start = time.time()

    for categories in paths:
        for category in categories:
            for image in category:
                cv2_shape = get_image_resolution(image, "cv2_shape")
                if cv2_shape > largest_shape:
                    largest_shape = cv2_shape
                    largest_shape_path = image
                elif cv2_shape < smallest_shape:
                    smallest_shape = cv2_shape
                    smallest_shape_path = image

    stop = time.time()

    largest_stats = [largest_shape, largest_shape_path]
    smallest_stats = [smallest_shape, smallest_shape_path]

    print(f"{'-' * 85}")
    print(f">>> Time Taken to Scan: {stop - start} seconds")
    print(f">>> Largest Shape: {largest_shape}")
    print(f"Largest Image at '{largest_shape_path}'")
    print(f">>> Smallest Shape: {smallest_shape}")   
    print(f"Smallest Image at '{smallest_shape_path}'") 
    print(f"{'-' * 85}")

    return largest_stats, smallest_stats

def display_size_comparisons():
    print(f"{'-' * 85}")
    print(f">>> Largest Shape: (2713, 2517, 3)")
    print(f"Largest Image at 'C:/Users/harsh/Desktop/Python/Projects/Pneumonia-Detection/dataset/test/NORMAL/NORMAL2-IM-0030-0001.jpeg'")
    print(f"{'-' * 65}")
    print(f">>> Smallest Shape: (127, 384, 3)")   
    print(f"Smallest Image at 'C:/Users/harsh/Desktop/Python/Projects/Pneumonia-Detection/dataset/train/PNEUMONIA/person407_virus_811.jpeg'") 
    print(f"{'-' * 85}")

# display_image_comparisons()
# show_image_counts([train_dir, test_dir, val_dir])

