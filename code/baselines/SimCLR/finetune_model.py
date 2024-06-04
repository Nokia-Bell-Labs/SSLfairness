# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import csv
import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# Library scripts
import raw_data_processing
import data_pre_processing
import simclr_models
import simclr_utitlities
import transformations

# from definitions import ROOT_DIR


# MIMIC
# working_directory = 'MIMIC/'
# data_folder = 'MIMIC'  # data to fine-tune
# input_shape = (48, 76)
# output_shape = 2  # edit this to be the number of label classes
# monitor = "val_recall"
# # pre-trained model
# pretrained_model_name = f'{seed}_simclr_200_scale_negate.hdf5'

working_directory = 'MESA/'
data_folder = 'MESA'  # data to fine-tune
input_shape = (101, 5)
output_shape = 2  # edit this to be the number of label classes
monitor = "val_auc"
pretrained_model_name = f'{seed}_simclr_200_scale_negate.hdf5'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Sleep EEG
# working_directory = 'SleepEEG/'
# data_folder = 'SleepEEG'
# input_shape = (178, 1)
# output_shape = 5

dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)

# SIMCLR finetuning
total_epochs = 100
batch_size = 128
added_layers = 2
hidden_size = 128
early_stopping = False
weighted = True
resampling = False
# uncomment for the server only
# print("Current working directory: {}".format(os.getcwd()))
# os.chdir(os.path.join("code", "baselines", "SimCLR"))
# print("Current working directory: {}".format(os.getcwd()))

# uncomment following 2 lines for server
os.chdir(os.path.join('code', 'baselines', 'SimCLR'))
print("CWD: {}".format(os.getcwd()))

# Load preprocessed data
if resampling:
    np_train = (np.load(os.path.join(data_folder, 'train_resampled_x.npy')),
                np.load(os.path.join(data_folder, 'train_resampled_y.npy')))
else:
    np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
                np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

# LOGGING
logdir = os.path.join("../../experiments_logs", "{}_200_{}_e{}_l{}_hs{}_es{}_bs{}_w{}_r{}"
                      .format(data_folder, start_time_str, total_epochs, added_layers, hidden_size, early_stopping, batch_size, weighted, resampling))
print("Log directory: {}".format(logdir))
if not os.path.exists(logdir):
    print("Creating log directory...")
    os.makedirs(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
# pretrained_model = simclr_models.attach_simclr_head(base_model)

pretrained_model = tf.keras.models.load_model(os.path.join(working_directory, pretrained_model_name))

linear_evaluation_model = simclr_models.create_linear_model_from_base_model(pretrained_model, output_shape=output_shape,
                                                                            hidden_size=hidden_size, added_layers=added_layers)

subfolder = "{}_200_l{}_hs{}_e{}_es{}_bs{}_w{}_r{}".format(start_time_str, added_layers, hidden_size, total_epochs, early_stopping, batch_size, weighted, resampling)
linear_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.finetuned.{val_recall:.2f}.hdf5")
if not os.path.exists(os.path.join(data_folder, subfolder)):
    os.makedirs(os.path.join(data_folder, subfolder))

# linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_finetuned_l{added_layers}_hs{hidden_size}_e{total_epochs}_es{early_stopping}_bs{batch_size}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
                                                         monitor=monitor, mode='max', save_best_only=True,
                                                         save_weights_only=False, verbose=0,
                                                         )
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
if weighted:
    class_weight = simclr_utitlities.get_class_weigths(np_train)
    # class_weight = {0: 0.58, 1: 3}


# Early-stopping to avoid overfitting
if early_stopping:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=5)
    if weighted:
        training_history = linear_evaluation_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = linear_evaluation_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
        )
else:
    if weighted:
        training_history = linear_evaluation_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = linear_evaluation_model.fit(
            x=np_train[0],
            y=np_train[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=total_epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
        )

# best model
best_val_recall = '{:.2f}'.format(round(max(training_history.history[monitor]), 2))
linear_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.finetuned.{}.hdf5".format(best_val_recall))
print("Loading file from: {}".format(linear_eval_best_model_file_name))

linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)
print("Model with the highest validation recall")
metrics = simclr_utitlities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
print(metrics)
# Write metrics to file
with open(linear_eval_best_model_file_name.replace('.hdf5', '.csv'), 'w') as f:
    w = csv.writer(f)
    metrics.pop("Confusion Matrix")
    w.writerow(metrics.keys())
    w.writerow(metrics.values())

# print("Model with the highest validation AUC:")
# print(simclr_utitlities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Model in the last epoch")
print(simclr_utitlities.evaluate_model_simple(linear_evaluation_model.predict(np_test[0]), np_test[1], return_dict=True))
