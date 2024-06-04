# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import datetime

import tensorflow as tf
import numpy as np

import os
import simclr_models
import simclr_utitlities

seed = 2
tf.random.set_seed(seed)
np.random.seed(seed)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# SIMCLR finetuning
# to run
# epochs 200 with early stopping (base model 100) starts at 14/11 11:05 f2_fl
# epochs 200 with early stopping (base model 200 dr01)
# epochs 100 without early stopping (base model 100) starts at 14/11 11:08 f2_fl
# epochs 100 without early stopping (base model 200 dr01) starts at 14/11 11:23 f2_fl
total_epochs = 200
# batch_size = 16
# added_layers = 2
# hidden_size = 128
early_stopping = True
weighted = True
batch_size = 128  # in the finetuning code, it's 16
monitor = 'val_loss'
mode = 'min'

# MIMIC and its pretrained model
# dataset = MIMIC
# working_directory = 'MIMIC/'
# data_folder = 'MIMIC'  # data to fine-tune
# pretrained_model_name = f'{seed}_simclr_200_scale_negate.hdf5'
# model_path = os.path.join(working_directory, pretrained_model_name)
# input_shape = (48, 76)
# output_shape = 2  # edit this to be the number of label classes

# server
dataset = 'MESA'
working_directory = 'code/baselines/SimCLR/MESA/'
data_folder = 'code/baselines/SimCLR/MESA'  # data to fine-tune
# local
# working_directory = 'MESA/'
# data_folder = 'MESA'  # data to fine-tune
input_shape = (101, 5)
output_shape = 2  # edit this to be the number of label classes

#  motionsense
# dataset = 'motionsense'
# working_directory = 'code/baselines/SimCLR/motionsense/'
# data_folder = 'code/baselines/SimCLR/motionsense'  # data to fine-tune
# input_shape = (400, 3)  # (samples, 400, 3)
# output_shape = 2  # edit this to be the number of label classes

# GLOBEM
# dataset = 'GLOBEM'
# working_directory = 'code/baselines/SimCLR/GLOBEM/'
# data_folder = 'code/baselines/SimCLR/GLOBEM'  # data to fine-tune
# input_shape = (28, 1390)  # (samples, 400, 3)
# output_shape = 2  # edit this to be the number of label classes

# GLOBEM
# pretrained_model_name = f'{seed}_simclr_200_scale_negate_dr0.1.hdf5'
# All rest
pretrained_model_name = f'{seed}_simclr_200_scale_negate.hdf5'
model_path = os.path.join(working_directory, pretrained_model_name)


# Load preprocessed data
np_train = (np.load(os.path.join(data_folder, 'train_x.npy')),
            np.load(os.path.join(data_folder, 'train_y.npy')))
np_val = (np.load(os.path.join(data_folder, 'val_x.npy')),
          np.load(os.path.join(data_folder, 'val_y.npy')))
np_test = (np.load(os.path.join(data_folder, 'test_x.npy')),
           np.load(os.path.join(data_folder, 'test_y.npy')))


def train_model_freezing_alternatives(frozen_layers, first_last=False, middle=False):
    # LOGGING
    # Uncomment next line for server
    # os.chdir(os.path.join('code', 'baselines', 'SimCLR'))
    print("Current working directory: {}".format(os.getcwd()))

    start_time = datetime.datetime.now()
    start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
    tf.keras.backend.set_floatx('float32')

    if first_last:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}_fl"
                          .format(dataset, start_time_str, total_epochs, early_stopping, batch_size, weighted, frozen_layers))
    elif middle:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}_m"
                              .format(dataset, start_time_str, total_epochs, early_stopping,
                                      batch_size, weighted, frozen_layers))
    else:
        logdir = os.path.join("code", "experiments_logs", "{}_{}_e{}_es{}_bs{}_w{}_fr{}"
                              .format(dataset, start_time_str, total_epochs, early_stopping,
                                      batch_size, weighted, frozen_layers))

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # adds 2 dense layers (similarly to finetune_model file, and freezes #frozen_layers
    full_evaluation_model = simclr_models.create_full_classification_model_from_base_model(simclr_model, output_shape,
                                                                                           model_name="TPN",
                                                                                           intermediate_layer=7,
                                                                                           frozen_layers=frozen_layers,
                                                                                           first_last=first_last,
                                                                                           middle=middle)

    print(full_evaluation_model.summary())
    if first_last:
        subfolder = "{}_e{}_es{}_bs{}_w{}_f{}_fl".format(start_time_str, total_epochs,
                                                  early_stopping, batch_size, weighted, frozen_layers)
    elif middle:
        subfolder = "{}_e{}_es{}_bs{}_w{}_f{}_m".format(start_time_str, total_epochs,
                                                         early_stopping, batch_size, weighted, frozen_layers)
    else:
        subfolder = "{}_e{}_es{}_bs{}_w{}_f{}".format(start_time_str, total_epochs,
                                                         early_stopping, batch_size, weighted, frozen_layers)

    # TODO edit for changing metric
    full_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.frozen.{val_loss:.2f}.hdf5")
    if not os.path.exists(os.path.join(data_folder, subfolder)):
        os.makedirs(os.path.join(data_folder, subfolder))

    best_model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=full_eval_best_model_file_name,
                                                             monitor=monitor, mode=mode, save_best_only=True,
                                                             save_weights_only=False, verbose=1
                                                             )

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    if weighted:
        class_weight = simclr_utitlities.get_class_weigths(np_train)
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10)
            training_history = full_evaluation_model.fit(
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
            training_history = full_evaluation_model.fit(
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
        if early_stopping:
            es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=10)
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback, es_callback],
                validation_data=np_val,
            )
        else:
            training_history = full_evaluation_model.fit(
                x=np_train[0],
                y=np_train[1],
                batch_size=batch_size,
                shuffle=True,
                epochs=total_epochs,
                callbacks=[best_model_callback, tensorboard_callback],
                validation_data=np_val,
            )

    full_evaluation_model.evaluate(
        x=np_test[0],
        y=np_test[1],
        return_dict=True
    )

    # best model
    # TODO edit min/max for changing metric
    best_val_recall = '{:.2f}'.format(round(min(training_history.history[monitor]), 2))
    full_eval_best_model_file_name = os.path.join(data_folder, subfolder, "simclr.frozen.{}.hdf5".format(best_val_recall))

    full_eval_best_model = tf.keras.models.load_model(full_eval_best_model_file_name)
    # print("\"Supervised\" Model\n{}".format(full_eval_best_model.summary()))

    print("Model with {} {} ():".format(mode, monitor, full_eval_best_model_file_name))
    print(
        simclr_utitlities.evaluate_model_simple(full_eval_best_model.predict(np_test[0]), np_test[1], return_dict=True))
    print("Model in last epoch")
    print(simclr_utitlities.evaluate_model_simple(full_evaluation_model.predict(np_test[0]), np_test[1],
                                                  return_dict=True))

# load simclr base model
simclr_model = tf.keras.models.load_model(model_path)
train_model_freezing_alternatives(frozen_layers=2, first_last=True)

# re-read original base model
simclr_model = tf.keras.models.load_model(model_path)
train_model_freezing_alternatives(frozen_layers=1, middle=True)

for frozen in range(4):
    simclr_model = tf.keras.models.load_model(model_path)
    train_model_freezing_alternatives(frozen_layers=frozen)

