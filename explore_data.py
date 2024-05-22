#%%
import h5py
import scipy.io
import numpy as np
#%%
#We need I0 (image), S0 (segmentation map), and text description d
text = scipy.io.loadmat('./data/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')

ind = scipy.io.loadmat('./data/benchmark/ind.mat')
language = scipy.io.loadmat('./data/benchmark/language_original.mat')

G1 = h5py.File('./data/supervision_signals/G1.h5', 'r')

# %%

print(G1.keys())

for key in G1.keys():
    print(key)
    print(G1[key].shape)
    print(np.unique(np.array(G1[key])[0]))
# %%
print (text.keys())
text = text['hn2']
#%%
print(text.shape)
print(G1['b_'].shape)

#For Gshape I need segmentation map and text description
# %%
print(ind.keys())
print(ind['train_ind'])

# %%

print(language.keys())
print(language['engJ'][0])
print(language['cate_new'][0])
print(language['color_'][0])
print(language['gender_'][0])
print(language['sleeve_'][0])
print(language['codeJ'][0])

# %%
