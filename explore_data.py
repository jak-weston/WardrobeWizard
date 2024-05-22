#%%
import h5py
import scipy.io

#We need I0 (image), S0 (segmentation map), and text description d
text = scipy.io.loadmat('./data/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')

G1 = h5py.File('./data/supervision_signals/G1.h5', 'r')

# %%

print(G1.keys())

for key in G1.keys():
    print(key)
    print(G1[key])
# %%
print (text.keys())
text = text['hn2']
#%%
print(text.shape)
print(G1['b_'].shape)

#For Gshape I need segmentation map and text description
# %%
