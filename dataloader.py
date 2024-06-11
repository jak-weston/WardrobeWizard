import h5py
import scipy.io
import torch
import numpy as np
import pandas as pd

# FOR DEEPFASHION DATASET
# def get_text_data():
#     text = scipy.io.loadmat('./data/encode_hn2_rnn_100_2_full.mat')
#     text = text['hn2'] #100 dimensions containing something
#     return torch.tensor(np.array(text, dtype=np.float32))

# def load_segmented_images():
#     G2 = h5py.File('./data/G2.h5', 'r')
#     return torch.tensor(np.array(G2['b_']))

# def load_full_images():
#     G2 = h5py.File('./data/G2.h5', 'r')
#     return torch.tensor(np.array(G2['ih']))

# def get_image_mean():
#     G2 = h5py.File('./data/G2.h5', 'r')
#     return torch.tensor(np.array(G2['ih_mean']))

# def get_raw_text_data():
#     text = scipy.io.loadmat('./data/language_original.mat')
#     return text['engJ']

# FOR PET DATASET
def get_text_data():
    meta = pd.read_csv('./data/list.txt', sep=' ')
    pet_type = meta['class'].values #There are 37 classes
    pet_type = pd.get_dummies(pet_type)
    return torch.tensor(pet_type.values)

def load_segmented_images():
    return torch.tensor(np.load('./data/segmentations.npy'))

def load_full_images():
    return torch.tensor(np.load('./data/images.npy')).permute(0, 3, 1, 2)  

def get_image_mean():
    return torch.zeros((3, 128, 128))


def get_raw_text_data():
    meta = pd.read_csv('./data/list.txt', sep=' ')
    pet_class_map = {
        1: 'Abyssinian',
        2: 'american_bulldog',
        3: 'american_pit_bull_terrier',
        4: 'basset_hound',
        5: 'beagle',
        6: 'Bengal',
        7: 'Birman',
        8: 'Bombay',
        9: 'boxer',
        10: 'British_Shorthair',
        11: 'chihuahua',
        12: 'Egyptian_Mau',
        13: 'english_cocker_spaniel',
        14: 'english_setter',
        15: 'german_shorthaired',
        16: 'great_pyrenees',
        17: 'havanese',
        18: 'japanese_chin',
        19: 'keeshond',
        20: 'leonberger',
        21: 'Maine_Coon',
        22: 'miniature_pinscher',
        23: 'newfoundland',
        24: 'Persian',
        25: 'pomeranian',
        26: 'pug',
        27: 'Ragdoll',
        28: 'Russian_Blue',
        29: 'saint_bernard',
        30: 'samoyed',
        31: 'scottish_terrier',
        32: 'shiba_inu',
        33: 'Siamese',
        34: 'Sphynx',
        35: 'staffordshire_bull_terrier',
        36: 'wheaten_terrier',
        37: 'yorkshire_terrier'
    }
    pet_class = meta['class'].map(pet_class_map)
    return pet_class.values