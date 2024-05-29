import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from scipy.io import loadmat
from scipy.io import savemat

# Load the .mat file
mat = loadmat('data/language_original.mat')

# Access the variables through the dictionary
for k, v in mat.items():
    globals()[k] = v

print(mat.keys())

# Define constants
dim_voc = 539
bsz = 1
m = 78979
dim_h = 100
dim_cate_new = 19
dim_color = 17
dim_gender = 2
dim_sleeve = 4
num_layers = 2

# Create data tensors
data_cate_new = torch.IntTensor(m, 1)
data_color = torch.IntTensor(m, 1)
data_gender = torch.IntTensor(m, 1)
data_sleeve = torch.IntTensor(m, 1)
for i in range(m):
    data_cate_new[i][0] = int(cate_new[i][0] - 1)
    data_color[i][0] = int(color_[i][0] - 1)
    data_gender[i][0] = int(gender_[i][0])
    data_sleeve[i][0] = int(sleeve_[i][0] - 1)

# Check if GPU is available or if we should use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define the network
class define_network(nn.Module):
    def __init__(self):
        super(define_network, self).__init__()
        self.rnn = nn.RNN(dim_voc, dim_h, num_layers)
        self.net_cate_new = nn.Linear(dim_h, dim_cate_new)
        self.net_color = nn.Linear(dim_h, dim_color)
        self.net_gender = nn.Linear(dim_h, dim_gender)
        self.net_sleeve = nn.Linear(dim_h, dim_sleeve)

    def forward(self, x):
        h0 = Variable(torch.zeros(num_layers, bsz, dim_h).to(device))
        _, hn = self.rnn(x, h0)
        hn2 = hn[-1]
        y_cate_new = self.net_cate_new(hn2)
        y_color = self.net_color(hn2)
        y_gender = self.net_gender(hn2)
        y_sleeve = self.net_sleeve(hn2)
        return hn2, y_cate_new, y_color, y_gender, y_sleeve

model = define_network().to(device)

# Load the model state
model.load_state_dict(torch.load('data/rnn_latest.pth', map_location=device))

# Define the criterion
criterion = nn.CrossEntropyLoss().to(device)

# Create label variables
cuda_label_cate_new = Variable(torch.LongTensor(bsz).zero_().to(device))
cuda_label_color = Variable(torch.LongTensor(bsz).zero_().to(device))
cuda_label_gender = Variable(torch.LongTensor(bsz).zero_().to(device))
cuda_label_sleeve = Variable(torch.LongTensor(bsz).zero_().to(device))

model.eval()

# Initialize result arrays
test_hn2 = np.zeros((m, dim_h))
test_cate_new = np.zeros((m, dim_cate_new))
test_color = np.zeros((m, dim_color))
test_gender = np.zeros((m, dim_gender))
test_sleeve = np.zeros((m, dim_sleeve))

# Process each sample
for sample_id in range(m):
    if sample_id % 1000 == 1:
        print(sample_id)
    c = codeJ[sample_id][0]
    l = len(c)
    cuda_c_onehot = torch.zeros(l, bsz, dim_voc).to(device)
    for i in range(l):
        cuda_c_onehot[i][0][int(c[i][0]-1)] = 1
    cuda_c_onehot = Variable(cuda_c_onehot)

    hn2, y_cate_new, y_color, y_gender, y_sleeve = model(cuda_c_onehot)
    test_hn2[sample_id] = hn2.data[0].cpu().numpy()
    test_cate_new[sample_id] = y_cate_new.data[0].cpu().numpy()
    test_color[sample_id] = y_color.data[0].cpu().numpy()
    test_gender[sample_id] = y_gender.data[0].cpu().numpy()
    test_sleeve[sample_id] = y_sleeve.data[0].cpu().numpy()

# Save the results
result = {"hn2": test_hn2}
savemat("data/encode_hn2_rnn_100_2_full.mat", result)
