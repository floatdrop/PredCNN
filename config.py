class video_pixel_network_config:
    input_shape = [64, 64, 1]

    # RMB config
    rmb_c = 64

    # Encoder config
    encoder_rmb_num = 4
    encoder_rmb_dilation = True
    encoder_rmb_dilation_scheme = [1, 2, 4, 8, 1, 2, 4, 8]

    # Decoder config
    decoder_rmb_num = 6

    # Training config
    epochs_num = 50000
    iters_per_epoch = 1 # 450
    truncated_steps = 6
    learning_rate = 1e-4

    # Data config
    train_sequences_num = 7000

    # tensorflow config
    max_to_keep = 3
    test_every = 100
