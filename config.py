config = {}
# Configurations for deepfashion

# # learning batch
# config['batchSize'] = 4
# config['win_size'] = 128
# config['lr_win_size'] = 8


# # specific size
# config['n_map_all'] = 7
# config['n_condition'] = 4

# config['n_c'] = 3

# config['nz'] = 80
# config['nt_input'] = 100
# config['nt'] = 20
# config['lambda_fake'] = 0.9
# config['lambda_mismatch'] = 1 - config['lambda_fake']

# config['lambda_real']= 1


# # Displaying and logging
# config['resume_iter'] = 0
# config['disp_win_id'] = 0

# Configurations for pet dataset

# learning batch
config['batchSize'] = 4
config['win_size'] = 128
config['lr_win_size'] = 8


# specific size
config['n_map_all'] = 3
# config['n_condition'] = 4

config['n_c'] = 3

config['nz'] = 80
config['nt_input'] = 37
config['nt'] = 20
config['lambda_fake'] = 0.999
config['lambda_mismatch'] = 1 - config['lambda_fake']

config['lambda_real']= 1


# Displaying and logging
config['resume_iter'] = 0
config['disp_win_id'] = 0