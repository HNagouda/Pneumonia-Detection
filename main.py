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


# ======= Visualization ========
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

# ======= Images ========
from VisualizerClass import Visualizer
import cv2
from glob import glob
from PIL import Image
import albumentations as alb

# ========= SKLearn =========
from skimage.transform import resize
from sklearn.model_selection import train_test_split

# ======== Transfer Learning ======
from efficientnet.tfkeras import *
from tensorflow.keras.applications import ResNet50, Xception

# ======== TensorFlow ========
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from tensorflow.python.keras.layers.advanced_activations import *
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.constraints import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from tensorflow.keras.metrics import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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

base_path = "C:/Users/harsh/Desktop/Python/Projects/222 - Pneumonia-Detection/"

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
        print(f">>> Counts in {dir.split(sep='/')[-1].upper()} SET:")
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


# -----------------------------------------------------------------------------------------------------------
# ============================================= DATA GENERATOR ==============================================
# -----------------------------------------------------------------------------------------------------------

def get_generator(gen_type, augmentations, use_augmentations):
    if gen_type == "train":
        if use_augmentations:
            datagen = ImageDataGenerator(**augmentations)
        else:
            datagen = ImageDataGenerator()

    elif gen_type == "test" or gen_type == "val":
        datagen = ImageDataGenerator()
    
    return datagen

def call_and_define_generators(use_augmentations, batch_size, target_size):

    print(f"\n{'-' * 85}")
    print(f">>> PREPARING IMAGE-GENERATORS...")

    train_path = os.path.join(base_path, "dataset/train/")
    test_path = os.path.join(base_path, "dataset/test/")
    val_path = os.path.join(base_path, "dataset/val/")   

    train_datagen = get_generator("train", augmentations, use_augmentations)
    test_datagen = get_generator("test", augmentations, use_augmentations)
    val_datagen = get_generator("val", augmentations, use_augmentations)

    train_datagen = train_datagen.flow_from_directory(
        train_path,
        target_size,
        batch_size = batch_size,
        color_mode = "grayscale",
        class_mode = "binary"
    )
    
    test_datagen = test_datagen.flow_from_directory(
        test_path,
        target_size,
        batch_size = batch_size,
        color_mode = "grayscale",
        class_mode = "binary"   
    )

    val_datagen = val_datagen.flow_from_directory(
        val_path,
        target_size,
        batch_size = batch_size,
        color_mode = "grayscale",
        class_mode = "binary"
    )

    print(f">>> IMAGE-GENERATORS SUCCESSFULLY CREATED")
    print(f"\n{'-' * 85}")

    return [train_datagen, test_datagen, val_datagen]


# -----------------------------------------------------------------------------------------------------------
# =========================================== MODEL CALLBACKS ===============================================
# -----------------------------------------------------------------------------------------------------------

def reducelronplateau():
    reducelronplateau = ReduceLROnPlateau(
        monitor='loss', factor=0.05, 
        patience=5, verbose=1, mode='auto',
        min_delta=0.0001, cooldown=0, min_lr=0.001)

    return reducelronplateau

def tensorboard(logs_dir):
    tensorboard = TensorBoard(
        log_dir="Tensorboard_Logs", histogram_freq=0, 
        write_graph=True, write_images=False, 
        update_freq='epoch', profile_batch=2, 
        embeddings_freq=0, embeddings_metadata=None)

    return tensorboard

def modelcheckpoint(checkpoint_filepath):
    modelcheckpoint = ModelCheckpoint(
        filepath=checkpoint_filepath, 
        save_weights_only=True, save_best_only=True,
        monitor='val_acc', mode='max')
    
    return modelcheckpoint

def earlystopping():    
    earlystopping = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=0, 
        verbose=0, mode='auto', baseline=None, 
        restore_best_weights=False)
    
    return earlystopping

def get_model_callbacks(tensorboard_logs_dir, model_checkpoint_filepath):
    callbacks = [
        reducelronplateau(),
        tensorboard(tensorboard_logs_dir),
        modelcheckpoint(model_checkpoint_filepath),
        earlystopping()
    ]

    return callbacks


# -----------------------------------------------------------------------------------------------------------
# ============================================= MODEL OPTIMIZERS ============================================
# -----------------------------------------------------------------------------------------------------------

def get_mixed_precision_opt(optimizer):
    return tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)


# -----------------------------------------------------------------------------------------------------------
# ============================================ TRANSFER LEARNING ============================================
# -----------------------------------------------------------------------------------------------------------

# ================================= LOADING ALL MODELS =================================
# =================  EfficientNet(s) B0-B7 ================= 

def call_transfer_models(input_shape):

    transfer_models_list = []

    input_tensor = Input(shape=input_shape)

    efn0 = EfficientNetB0(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn1 = EfficientNetB1(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn2 = EfficientNetB2(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn3 = EfficientNetB3(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn4 = EfficientNetB4(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn5 = EfficientNetB5(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn6 = EfficientNetB6(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)
    efn7 = EfficientNetB7(input_shape=input_shape, weights='imagenet', 
                          include_top=False, input_tensor=input_tensor)

    # ======================== Xception ========================
    xception = Xception(input_shape=input_shape, weights='imagenet', 
                        include_top=False, input_tensor=input_tensor)

    # ======================== ResNet50 ========================
    resnet50 = ResNet50(input_shape=input_shape, weights='imagenet', 
                        include_top=False, input_tensor=input_tensor)

    transfer_models_list = [efn0, efn1, efn2, efn3, efn4, efn5, efn6, efn7, xception, resnet50]

    return transfer_models_list

# ================================ ADDING INPUT LAYERS =================================

def add_output_layers(transfer_models_list):
    for i, model_name in enumerate(transfer_models_list):
        x = model_name.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(250, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(100, activation='relu')(x)
        predictions = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=model_name.input, outputs=predictions)
        transfer_models_list[i] = model

    return transfer_models_list


# -----------------------------------------------------------------------------------------------------------
# ======================================= IMAGE CLASSIFICATION MODELS =======================================
# -----------------------------------------------------------------------------------------------------------


# def yeetModel(inputShape):
#     return RIPyeetedmodel


# =================================== MODEL COMPILING ===================================

def compile_model(model, enable_mixed_precision):
    # optimizer 
    optimizer = Adam(
        learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, 
        epsilon=epsilon, amsgrad=amsgrad, name='Adam'
    )
    if enable_mixed_precision:
        optimizer = get_mixed_precision_opt(optimizer)
    else:
        pass
        
    # model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# =================================== MODEL TRAINING ====================================

def train_model(model, epochs, use_callbacks):

    callbacks = get_model_callbacks(tensorboard_logs_dir, model_checkpoint_filepath)

    start = time.time()

    if use_callbacks:
        history = model.fit(
            train_generator,
            epochs = epochs,
            steps_per_epoch = steps_per_epoch,
            callbacks = callbacks,
            validation_data = test_generator,
            validation_steps = validation_steps    
        )

    else:
        history = model.fit(
        train_generator,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        validation_data = test_generator,
        validation_steps = validation_steps    
    )

    stop = time.time()
    total_training_time = stop - start

    return history, total_training_time

# ================================= TRAINING ANALYSIS ===================================

def export_model_stats(model_history, total_training_time, plot_path): 
    plot_title = f"""Visualizing Model Progress
    Total Training Time: {total_training_time} seconds
    """

    history = model_history.history


    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=['Loss', 'Accuracy'])

    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['val_loss'],
                            mode='lines+markers', name='Loss'), 
                            row=1, col=1)

    fig.add_trace(go.Scatter(x=np.arange(1, 11), y=history['val_binary_accuracy'],
                            mode='lines+markers', name='Accuracy'), 
                            row=1, col=2)

    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_xaxes(title_text='Epochs', row=1, col=2)
        
    fig.update_layout(title=plot_title)
    fig.write_image(plot_path) 


def save_model(trained_model, models_dir, model_name): 
    trained_model = trained_model
    trained_model.save(f"{models_dir}/{model_name}.hdf5")


def load_saved_model(models_dir, model_name):
    model = load_model(f"{models_dir}/{model_name}.hdf5", custom_objects={
                              'dice_coef_loss': dice_coef_loss, 
                              'iou': iou, 
                              'dice_coef': dice_coef
                          })

    return model


# =================================== MODEL EVALUTION ===================================
def evaluate_model(data_generator, models_dir, saved_model_name):
    saved_model = load_saved_model(models_dir, saved_model_name)

    scores = saved_model.evaluate(data_generator, steps=len(test_set)/batch_size)
    
    return scores

def print_scores(evaluated_scores):
    loss, accuracy = evaluated_scores

    print(f"""
    Evaluation Loss: {loss}
    Evaluation Accuracy: {accuracy}
    """)


# -----------------------------------------------------------------------------------------------------------
# ================================= PATHS, PARAMETERS, AND MODEL TUNING =====================================
# -----------------------------------------------------------------------------------------------------------
# RUNTIME NAME ***
runtime_name = "EfficientNet B0 - Trial 1"  # MUST change this every time the code is run

# Model Hyperparameters
EPOCHS = 10
batch_size = 32
learning_rate = 0.0001
input_shape = (256, 256, 3)
target_size = (256, 256)

# steps_per_epoch = len(train_set) / batch_size
# validation_steps = len(val_set) / batch_size

steps_per_epoch = 2
validation_steps = 2

# Data-Generator Parameters
use_augmentations = True
use_callbacks = False
enable_mixed_precision = True
model_checkpoint_filepath = f"{base_path}/Model_Checkpoints/{runtime_name}_checkpoint.hdf5"
tensorboard_logs_dir = f"{base_path}/Tensorboard_Logs"

# Data-Generator Augmentations
augmentations = dict(
    # rotation_range=0.25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    brightness_range=(0.45, 0.80),
    fill_mode='nearest',
    shear_range=0.1,
    zoom_range=0.5,
    horizontal_flip=True,
    # vertical_flip=True,
)

# Adam Optimizer arguments
beta_1, beta_2 = 0.9, 0.999
epsilon, amsgrad = 1e-07, False

# Loss and Metrics
loss = "binary_crossentropy"
metrics = ['binary_accuracy']

# Model Accuracy Findings
plot_model_stats_dir = f"{base_path}/Model_Stats_Plots"
plot_path = f"{plot_model_stats_dir}/{runtime_name}.jpg"

# Model Saving Paths
models_dir = f"{base_path}/Saved_Models"


# -----------------------------------------------------------------------------------------------------------
# ============================================ PROJECT EXECUTION ============================================
# -----------------------------------------------------------------------------------------------------------

# Image Sizes Scanned in the directories
show_image_counts(dataset_subdirs)
display_size_comparisons()

# ======================= GET GENERATORS ==========================
train_generator, test_generator, val_generator = call_and_define_generators(
                                    use_augmentations, batch_size, target_size
                                )


# ======================== PREPARING MODELS =======================
print(f"\n{'-' * 50} \nPREPARING MODELS... \n {'-' * 50}")
transfer_models_list = call_transfer_models(input_shape)
efn0, efn1, efn2, efn3, efn4, efn5, efn6, efn7, xception, resnet50 = add_output_layers(transfer_models_list)
print(f"\n{'-' * 50} \nALL MODELS SUCCESSFULLY PREPARED \n {'-' * 50}")

# ================== COMPILING & RUNNING THE MODEL ================

print(f"\n{'-' * 50} \nCOMPILING MODEL... \n {'-' * 50}")
Model = compile_model(efn0, enable_mixed_precision)
print(f"\n{'-' * 50} \nMODEL SUCCESSFULLY COMPILED \n {'-' * 50}")

print(f"\n{'-' * 50} \nBEGINNING MODEL TRAINING... \n {'-' * 50}")
Model_history = train_model(Model, EPOCHS, use_callbacks)
print(f"\n{'-' * 50} \nMODEL SUCCESSFULLY TRAINED \n {'-' * 50}")


# ==================== PLOTTING & SAVING MODEL ====================
save_model(trained_model=Model, models_dir=models_dir, model_name=runtime_name)
print(f"\n{'-' * 50} \nMODEL SAVED TO '{models_dir}/{runtime_name}.hdf5' \n {'-' * 50}")

export_model_stats(Model_history, plot_path)
print(f"\n{'-' * 50} \nPLOT OF MODEL PROGRESS SAVED AT {plot_path}")

# ======================= LOAD AND EVALUATE =======================
print(f"\n{'-' * 50} \nLOADING MODEL FOR EVALUATION... \n {'-' * 50}")
print(f"\n{'-' * 50} \nBEGINNING MODEL EVALUATION \n {'-' * 50}")

model_scores = evaluate_model(test_generator, models_dir, runtime_name)
print(f"{'-' * 50} \nMODEL SUCCESSFULLY EVALUATED ")

print(f"Model loss: {model_scores[0]}")
print(f"Model IOU: {model_scores[1]}")
print(f"Model Dice Coefficient: {model_scores[2]} \n{'-' * 50}")