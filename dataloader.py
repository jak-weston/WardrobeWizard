import h5py
import scipy.io

def get_text_data():
    text = scipy.io.loadmat('./data/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')
    text = text['hn2']
    return text

def get_binary_data():
    G1 = h5py.File('./data/supervision_signals/G1.h5', 'r')
    return G1['b_']