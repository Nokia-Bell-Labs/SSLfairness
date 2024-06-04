# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import csv
import os
import re
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd

# from keras import backend as K
# K.set_image_dim_ordering('th')
# from definitions import ROOT_DIR

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# switch to XLA for speed up
# tf.config.optimizer.set_jit(True)

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

# Libraries for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.manifold

sns.set_context('poster')

# Library scripts
import supervised_models_globem
import supervised_utilities

supervised_utilities.setup_system("0")

# Dataset-specific MIMIC
dataset = 'MIMIC'
working_directory = 'MIMIC/'
data_folder = '../SimCLR/MIMIC'
input_shape = (48, 76)  # (timesteps, channels)
output_shape = 2
monitor = 'val_loss'
mode = 'min'
PROTECTED_ATTRIBUTE = 'ETHNICITY'
MERGE_ON = 'SUBJECT_ID'
N_SAMPLES = 150

# Dataset-specific GLOBEM
# working_directory = 'GLOBEM/'
# data_folder = '../SimCLR/GLOBEM'
# input_shape = (28, 1390)  # (timesteps, channels)
# output_shape = 2
# monitor = 'val_loss'
# mode = 'min'

# Dataset-specific MESA
# working_directory = 'MESA/'
# data_folder = '../SimCLR/MESA'
# input_shape = (101, 5)  # (timesteps, channels)
# output_shape = 2
# monitor = 'val_loss'
# mode = 'min'

# Dataset-specific SleepEEG
# working_directory = 'SleepEEG/'
# data_folder = 'SleepEEG'
# input_shape = (178, 1)

def prepare_trainlistfile(train_listfile):
    # extract subject_id from stay
    regex = r"(?:^\d+)"
    train_listfile.loc[:, "SUBJECT_ID"] = train_listfile.stay.apply(lambda stay: re.search(regex, stay).group(0))
    train_listfile.SUBJECT_ID = train_listfile.SUBJECT_ID.astype(int)
    train_listfile.drop(['stay'], axis=1, inplace=True)
    return train_listfile

def scale_training_set(np_train, n_samples):
    print(os.getcwd())
    subjects = pd.read_csv(os.path.join('../../../', 'datasets', dataset, 'demographics_rich.csv'))
    train_listfile = pd.read_csv(os.path.join('../../../', 'datasets', dataset, 'train_listfile.csv'))
    train_listfile = prepare_trainlistfile(train_listfile)
    train_listfile = train_listfile.merge(subjects, how='left', on=MERGE_ON)

    features = []
    labels = []
    for attr in train_listfile[PROTECTED_ATTRIBUTE].value_counts().keys():
        print("Adding {} {} users to the training set...".format(n_samples, attr))
        try:
            features.append(np_train[0][train_listfile[PROTECTED_ATTRIBUTE] == attr][:n_samples])
            labels.append(np_train[1][train_listfile[PROTECTED_ATTRIBUTE] == attr][:n_samples])
        except Exception:
            print("Skipping {} users; not enough samples...".format(attr))
            continue

    # time.sleep(10)
    print("Features: {} - Labels: {}".format(np.concatenate(features, axis=0).shape, np.concatenate(labels, axis=0).shape))
    return (np.concatenate(features, axis=0), np.concatenate(labels, axis=0))

dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)
if not os.path.exists(os.path.join(working_directory, 'img')):
    os.mkdir(os.path.join(working_directory, 'img'))

# Load preprocessed data
# Uncomment next line for server
os.chdir(os.path.join('code', 'baselines', 'Supervised'))
print(os.getcwd())

np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
            np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))

print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

# SIMCLR training parameters
batch_size = 64
# decay_steps = 1000
added_layers = 2
epochs = 100
# temperature = 0.1
early_stopping = False
weighted = True

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')
# lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
# optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = supervised_utilities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

# Supervised Baseline Model
base_model = supervised_models_globem.run_supervised_base_model(input_shape, output_shape, dense_layers=added_layers, model_name="supervised_baseline_model_{}".format(added_layers))

print(base_model.summary())

print("Train - Validation - Test Set Shapes:")
print("Train X: {} - Val X: {} - Test X: {}".format(np_train[0].shape, np_val[0].shape, np_test[0].shape))
print("Train y: {} - Val y: {} - Test y: {}".format(np_train[1].shape, np_val[1].shape, np_test[1].shape))

# LOGGING
logdir = os.path.join("../../experiments_logs", "{}_{}_{}_supervised_e{}_l{}_es{}_bs{}_w{}_ss{}"
                      .format(working_directory.replace('/', ''), start_time_str, epochs, monitor, added_layers, early_stopping, batch_size, weighted, N_SAMPLES))
if not os.path.exists(logdir):
    os.makedirs(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

subfolder = "{}_l{}_e{}_es{}_bs{}_w{}_ss{}".format(start_time_str, added_layers, epochs, early_stopping, batch_size, weighted, N_SAMPLES)
# TODO edit for changing metric
linear_eval_best_model_file_name = os.path.join(working_directory, subfolder, "supervised.finetuned.{val_loss:.2f}.hdf5")
if not os.path.exists(os.path.join(working_directory, subfolder)):
    os.makedirs(os.path.join(working_directory, subfolder))

# linear_eval_best_model_file_name = f"{working_directory}{start_time_str}_finetuned_l{added_layers}_hs{hidden_size}_e{total_epochs}_es{early_stopping}_bs{batch_size}.hdf5"
best_model_callback = tf.keras.callbacks.ModelCheckpoint(linear_eval_best_model_file_name,
                                                         monitor=monitor, mode=mode, save_best_only=True,
                                                         save_weights_only=False, verbose=0,
                                                       )

#  sample training set
np_train_sampled = scale_training_set(np_train, N_SAMPLES)

# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
if weighted:
    class_weight = supervised_utilities.get_class_weigths(np_train)
    # class_weight = {0: 0.58, 1: 3}


# Early-stopping to avoid overfitting
if early_stopping:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10)
    if weighted:
        training_history = base_model.fit(
            x=np_train_sampled[0],
            y=np_train_sampled[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = base_model.fit(
            x=np_train_sampled[0],
            y=np_train_sampled[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback, es_callback],
            validation_data=np_val,
        )
else:
    if weighted:
        training_history = base_model.fit(
            x=np_train_sampled[0],
            y=np_train_sampled[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
            class_weight=class_weight
        )
    else:
        training_history = base_model.fit(
            x=np_train_sampled[0],
            y=np_train_sampled[1],
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            callbacks=[best_model_callback, tensorboard_callback],
            validation_data=np_val,
        )

# best model
# TODO edit min/max for changing metric
best_val = '{:.2f}'.format(round(min(training_history.history[monitor]), 2))
linear_eval_best_model_file_name = os.path.join(working_directory, subfolder, "supervised.finetuned.{}.hdf5".format(best_val))
print("Loading file from: {}".format(linear_eval_best_model_file_name))

linear_eval_best_model = tf.keras.models.load_model(linear_eval_best_model_file_name)
print("Model with the {} {}".format(mode, monitor))
metrics = supervised_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True)
print(metrics)
# Write metrics to file
with open(linear_eval_best_model_file_name.replace('.hdf5', '.csv'), 'w') as f:
    w = csv.writer(f)
    metrics.pop("Confusion Matrix")
    w.writerow(metrics.keys())
    w.writerow(metrics.values())

# print("Model with the highest validation AUC:")
# print(supervised_utilities.evaluate_model_simple(linear_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
print("Model in the last epoch")
print(supervised_utilities.evaluate_model_simple(base_model.predict(np_test[0]), np_test[1], return_dict=True))


# supervised_model_save_path = os.path.join(working_directory, f"{seed}_supervised_{epochs}_al{added_layers}.hdf5")
# print("Trained supervised model summary\n{}".format(trained_simclr_model.summary()))
# trained_simclr_model.save(simclr_model_save_path)

