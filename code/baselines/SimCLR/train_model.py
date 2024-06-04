import os
import pickle
import scipy
import datetime
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

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
import raw_data_processing
import data_pre_processing
import simclr_models
import simclr_utitlities
import transformations

# Dataset-specific MIMIC
# working_directory = 'MIMIC/'
# data_folder = 'MIMIC'
# input_shape = (48, 76)  # (timesteps, channels)

# Dataset-specific MESA
# working_directory = 'MESA/'
# data_folder = 'MESA'
# input_shape = (101, 5)  # (timesteps, channels)

# Dataset-specific SleepEEG
# working_directory = 'SleepEEG/'
# data_folder = 'SleepEEG'
# input_shape = (178, 1)

# MotionSense
# working_directory = 'motionsense/'
# data_folder = 'motionsense'
# input_shape = (400, 1390)  # (timesteps, channels)

# GLOBEM
working_directory = 'GLOBEM/'
data_folder = 'GLOBEM'
input_shape = (28, 1390)  # (timesteps, channels)

dataset_save_path = working_directory
if not os.path.exists(working_directory):
    os.mkdir(working_directory)
if not os.path.exists(os.path.join(working_directory, 'img')):
    os.mkdir(os.path.join(working_directory, 'img'))

# Load preprocessed data
# Uncomment next line for server
# os.chdir(os.path.join('code', 'baselines', 'SimCLR'))
# print(os.getcwd())

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
# pairs tried:
# batch size: 128 - epochs: 200
# batch size: 64 - epochs: 100
batch_size = 128
decay_steps = 1000
epochs = 200
temperature = 0.1
transform_funcs = [
    # transformations.rotation_transform_vectorized # Use rotation transformation -> developed for 3D
    transformations.scaling_transform_vectorized,
    transformations.negate_transform_vectorized
]
transform_funcs_str = "scale_negate"
transformation_function = simclr_utitlities.generate_composite_transform_function_simple(transform_funcs)

# trasnformation_indices = [2] # Use rotation trasnformation only
# trasnformation_indices = [1, 2] # Use Scaling and rotation trasnformation

# trasnform_funcs_vectorized = [
#     transformations.noise_transform_vectorized,
#     transformations.scaling_transform_vectorized,
#     transformations.rotation_transform_vectorized,
#     transformations.negate_transform_vectorized,
#     transformations.time_flip_transform_vectorized,
#     transformations.time_segment_permutation_transform_improved,
#     transformations.time_warp_transform_low_cost,
#     transformations.channel_shuffle_transform_vectorized
# ]
# transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled']

start_time = datetime.datetime.now()
start_time_str = start_time.strftime("%Y%m%d-%H%M%S")
tf.keras.backend.set_floatx('float32')

lr_decayed_fn = tf.keras.experimental.CosineDecay(initial_learning_rate=0.1, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.SGD(lr_decayed_fn)
# transformation_function = simclr_utitlities.generate_combined_transform_function(trasnform_funcs_vectorized, indices=trasnformation_indices)

# Base Model: A neural network base encoder, which is responsible for encoding the data samples into a latent space
base_model = simclr_models.create_base_model(input_shape, model_name="base_model")
# SimCLR Head: A projection head, which is effectively another neural network that projects the representations in the
# latent space into another space for contrastive learning
simclr_model = simclr_models.attach_simclr_head(base_model)
simclr_model.summary()

# LOGGING
# logdir = "../experiments_logs/" + start_time_str
print("Started training...")
trained_simclr_model, epoch_losses = simclr_utitlities.simclr_train_model(simclr_model, np_train[0], optimizer,
                                                                          batch_size, transformation_function,
                                                                          temperature=temperature, epochs=epochs,
                                                                          is_trasnform_function_vectorized=True,
                                                                          verbose=1)

simclr_model_save_path = os.path.join(working_directory, f"{seed}_simclr_{epochs}_{transform_funcs_str}_dr0.1.hdf5")
print("Trained SimCLR model summary\n{}".format(trained_simclr_model.summary()))
trained_simclr_model.save(simclr_model_save_path)

# plotting loss
plt.figure(figsize=(12, 8))
plt.plot(epoch_losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(os.path.join(working_directory, "img", "epoch_loss_plot_{}.png".format(start_time_str)))
