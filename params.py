"""Params for ADDA."""

# params for dataset and data loader
data_root = "data"
usps_data = data_root + '/USPS'
mnistm_data = data_root + '/MNIST_M'
svhn_data = data_root + '/SVHN'
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
batch_size = 50

# models
model_root = "snapshots"
src_encoder_restore = 'snapshots/{:s}/ADDA-source-encoder-best.pt'
src_classifier_restore = "snapshots/{:s}/ADDA-source-classifier-best.pt"
tgt_encoder_restore = "snapshots/{:s}/ADDA-target-encoder-best.pt"
d_model_restore = "snapshots/{:s}/ADDA-critic-best.pt"
src_model_trained = True
tgt_model_trained = True
d_input_dims = 500
d_hidden_dims = 500
d_output_dims = 2

# params for training network
num_gpu = 1
"""
num_epochs_pre = 100
log_step_pre = 100
save_step_pre = 10
eval_step_pre = 5

num_epochs_adapt = 2000
log_step_adapt = 100
save_step_adapt = 100
eval_step_adapt = 5
"""

num_epochs_pre = 20
log_step_pre = 100
save_step_pre = 10
eval_step_pre = 5

num_epochs_adapt = 40
log_step_adapt = 100
save_step_adapt = 20
eval_step_adapt = 5

manual_seed = None

# params for optimizing models
d_learning_rate = 1e-4
c_learning_rate = 1e-4
beta1 = 0.5
beta2 = 0.9
