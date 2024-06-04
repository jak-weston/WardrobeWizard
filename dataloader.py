import h5py
import scipy.io
import torch
import numpy as np

def get_text_data():
    text = scipy.io.loadmat('./data/encode_hn2_rnn_100_2_full.mat')
    text = text['hn2'] #100 dimensions containing something
    return torch.tensor(np.array(text, dtype=np.float32))

def load_segmented_images():
    G2 = h5py.File('./data/G2.h5', 'r')
    return torch.tensor(np.array(G2['b_']))

def load_full_images():
    G2 = h5py.File('./data/G2.h5', 'r')
    return torch.tensor(np.array(G2['ih']))

def get_image_mean():
    G2 = h5py.File('./data/G2.h5', 'r')
    return torch.tensor(np.array(G2['ih_mean']))

def get_raw_text_data():
    text = scipy.io.loadmat('./data/language_original.mat')
    return text['engJ']