# Configurations for the G_shape
config_gshape = {}

# learning batch
config_gshape['batchSize'] = 4
config_gshape['win_size'] = 128
config_gshape['lr_win_size'] = 8


# specific size
config_gshape['n_map_all'] = 7
config_gshape['n_condition'] = 4

config_gshape['nz'] = 80
config_gshape['nt_input'] = 100
config_gshape['nt'] = 20
config_gshape['lambda_fake'] = 0.9
config_gshape['lambda_mismatch'] = 1 - config_gshape['lambda_fake']


# Displaying and logging
config_gshape['resume_iter'] = 0
config_gshape['disp_win_id'] = 0