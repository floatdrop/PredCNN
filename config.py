input_shape = [192, 192, 1]

# RMB config
rmb_c = 64

# Encoder config
encoder_rmb_num = 4
encoder_rmb_dilation = False
encoder_rmb_dilation_scheme = [1, 2, 3, 4]

# Decoder config
decoder_rmb_num = 4

# Training config
epochs_num = 50000
iters_per_epoch = 100  # 450
truncated_steps = 4
learning_rate = 1e-4

# Data config
train_sequences_num = 7000

# tensorflow config
max_to_keep = 3
test_every = 10
