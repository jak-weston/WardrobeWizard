import h5py
import scipy.io
import torch
import numpy as np
def get_text_data():
    text = scipy.io.loadmat('./data/encode_hn2_rnn_100_2_full.mat')
    text = text['hn2'] #100 dimensions containing something
    return torch.tensor(np.array(text))

def load_segmented_images():
    G2 = h5py.File('./data/G2.h5', 'r')
    return torch.tensor(np.array(G2['b_']))

def get_text_data_jack():
    text = scipy.io.loadmat('data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')
    text = text['hn2'] #100 dimensions containing something
    return torch.tensor(np.array(text))

def load_segmented_images_jack():
    G2 = h5py.File('data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat', 'r')
    return torch.tensor(np.array(G2['b_']))