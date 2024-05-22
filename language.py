import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import randint

#############################
##   Initialize Variables ##
#############################
USE_GPU = False

vocabDim = 539
batchSize = 1
m = 78979
mTrain = 70000
hiddenDim = 100
categoryDim = 19
colorDim = 17
genderDim = 2
sleeveDim = 4
numLayers = 2
numClasses = 100
dtype = torch.float32

#############################
##      Training Model     ##
#############################

mat = sio.loadmat('data_release/benchmark/language_original.mat')
for k, v in mat.items():
    exec(k +  " = mat['" + k + "']")

mat = loadmat('data_release/benchmark/ind.mat')
trainInd = torch.IntTensor(mTrain)
for i in range(mTrain):
    trainInd[i] = int(mat['train_ind'][0][i] - 1)

dataCategoryNew = torch.IntTensor(m, 1)
dataColor = torch.IntTensor(m, 1)
dataGender = torch.IntTensor(m, 1)
dataSleeve = torch.IntTensor(m, 1)

for i in range(m):
    dataCategoryNew[i][0] = int(cate_new[i][0] - 1)
    dataColor[i][0] = int(color_[i][0] - 1)
    dataGender[i][0] = int(gender_[i][0])
    dataSleeve[i][0] = int(sleeve_[i][0] - 1)

class DefineNetwork(nn.Module):
    def __init__(self):
        super(DefineNetwork, self).__init__()
        self.rnn = nn.RNN(vocabDim, hiddenDim, numLayers)
        self.netCategoryNew = nn.Linear(hiddenDim, categoryDim)
        self.netColor = nn.Linear(hiddenDim, colorDim)
        self.netGender = nn.Linear(hiddenDim, genderDim)
        self.netSleeve = nn.Linear(hiddenDim, sleeveDim)

    def forward(self, x):
        h0 = Variable(torch.zeros(numLayers, batchSize, hiddenDim).cuda() if USE_GPU else torch.zeros(numLayers, batchSize, hiddenDim))
        _, hn = self.rnn(x, h0)
        hn2 = hn[-1]
        yCategoryNew = self.netCategoryNew(hn2)
        yColor = self.netColor(hn2)
        yGender = self.netGender(hn2)
        ySleeve = self.netSleeve(hn2)
        return hn2, yCategoryNew, yColor, yGender, ySleeve

model = DefineNetwork()
if USE_GPU:
    model.cuda()
criterion = nn.CrossEntropyLoss().cuda() if USE_GPU else nn.CrossEntropyLoss()
cudaLabelCategoryNew = Variable(torch.LongTensor(batchSize).zero_().cuda()) if USE_GPU else Variable(torch.LongTensor(batchSize).zero_())
cudaLabelColor = Variable(torch.LongTensor(batchSize).zero_()) if USE_GPU else Variable(torch.LongTensor(batchSize).zero_())
cudaLabelGender = Variable(torch.LongTensor(batchSize).zero_()) if USE_GPU else Variable(torch.LongTensor(batchSize).zero_())
cudaLabelSleeve = Variable(torch.LongTensor(batchSize).zero_()) if USE_GPU else Variable(torch.LongTensor(batchSize).zero_())

optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()
for iter in range(1000000 * 10):

    if iter == 50000:
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    assert batchSize == 1
    t = randint(0, USE_GPU-1)
    sampleId = trainInd[t]
    c = codeJ[sampleId][0]
    l = len(c)
    cudaCOnehot = torch.zeros(l, batchSize, vocabDim).cuda() if USE_GPU else torch.zeros(l, batchSize, vocabDim)
    for i in range(l):
        cudaCOnehot[i][0][int(c[i][0]-1)] = 1
    cudaCOnehot = Variable(cudaCOnehot)

    cudaLabelCategoryNew.data[0] = dataCategoryNew[sampleId][0]
    cudaLabelColor.data[0] = dataColor[sampleId][0]
    cudaLabelGender.data[0] = dataGender[sampleId][0]
    cudaLabelSleeve.data[0] = dataSleeve[sampleId][0]

    optimizer.zero_grad()
    hn2, yCategoryNew, yColor, yGender, ySleeve = model(cudaCOnehot)
    lossCategoryNew = criterion(yCategoryNew, cudaLabelCategoryNew)
    lossColor = criterion(yColor, cudaLabelColor)
    lossGender = criterion(yGender, cudaLabelGender)
    lossSleeve = criterion(ySleeve, cudaLabelSleeve)
    loss = lossCategoryNew + lossColor + lossGender + lossSleeve
    loss.backward()
    optimizer.step()

    if iter % 100 == 0:
        print('Training Iter %d: Loss = %.5f, category_new (%.5f), color (%.5f), gender(%.5f), sleeve(%.5f)' % (iter, loss.data[0], lossCategoryNew.data[0], lossColor.data[0], lossGender.data[0], lossSleeve.data[0]))

    if iter % 100000 == 1:
        torch.save(model.state_dict(), 'rnn_latest.pth')

#############################
##     testing model       ##
#############################

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.io import loadmat, savemat

# Define constants and data tensors
vocabDim = 539
batchSize = 1
m = 78979
hiddenDim = 100
categoryDim = 19
colorDim = 17
genderDim = 2
sleeveDim = 4
numLayers = 2
numClasses = 100
dtype = torch.float32

dataCategoryNew = torch.IntTensor(m, 1)
dataColor = torch.IntTensor(m, 1)
dataGender = torch.IntTensor(m, 1)
dataSleeve = torch.IntTensor(m, 1)

for i in range(m):
    dataCategoryNew[i][0] = int(categoryNew[i][0] - 1)
    dataColor[i][0] = int(color[i][0] - 1)
    dataGender[i][0] = int(gender[i][0])
    dataSleeve[i][0] = int(sleeve[i][0] - 1)

# Define the neural network model
class DefineNetwork(nn.Module):
    def __init__(self):
        super(DefineNetwork, self).__init__()
        self.rnn = nn.RNN(vocabDim, hiddenDim, numLayers)
        self.netCategoryNew = nn.Linear(hiddenDim, categoryDim)
        self.netColor = nn.Linear(hiddenDim, colorDim)
        self.netGender = nn.Linear(hiddenDim, genderDim)
        self.netSleeve = nn.Linear(hiddenDim, sleeveDim)

    def forward(self, x):
        h0 = Variable(torch.zeros(numLayers, batchSize, hiddenDim, dtype=dtype))
        if USE_GPU and torch.cuda.is_available():
            h0 = h0.cuda()
        _, hn = self.rnn(x, h0)
        hn2 = hn[-1]
        yCategoryNew = self.netCategoryNew(hn2)
        yColor = self.netColor(hn2)
        yGender = self.netGender(hn2)
        ySleeve = self.netSleeve(hn2)
        return hn2, yCategoryNew, yColor, yGender, ySleeve

# Load trained model
model = DefineNetwork()
if USE_GPU and torch.cuda.is_available():
    model.cuda()
    model.load_state_dict(torch.load('rnn_latest.pth'))
else:
    model.load_state_dict(torch.load('rnn_latest.pth', map_location=torch.device('cpu')))
model.eval()

# Define criterion
criterion = nn.CrossEntropyLoss()

# Initialize labels
cudaLabelCategoryNew = Variable(torch.LongTensor(batchSize).zero_())
cudaLabelColor = Variable(torch.LongTensor(batchSize).zero_())
cudaLabelGender = Variable(torch.LongTensor(batchSize).zero_())
cudaLabelSleeve = Variable(torch.LongTensor(batchSize).zero_())

# Process data and obtain results
testHn2 = np.zeros((m, hiddenDim))
testCategoryNew = np.zeros((m, categoryDim))
testColor = np.zeros((m, colorDim))
testGender = np.zeros((m, genderDim))
testSleeve = np.zeros((m, sleeveDim))

for sampleId in range(m):
    if sampleId % 1000 == 1:
        print(sampleId)

    c = codeJ[sampleId][0]
    l = len(c)
    cudaCOnehot = torch.zeros(l, batchSize, vocabDim, dtype=dtype)
    if USE_GPU and torch.cuda.is_available():
        cudaCOnehot = cudaCOnehot.cuda()

    for i in range(l):
        cudaCOnehot[i][0][int(c[i][0] - 1)] = 1

    cudaCOnehot = Variable(cudaCOnehot)
    hn2, yCategoryNew, yColor, yGender, ySleeve = model(cudaCOnehot)

    testHn2[sampleId] = hn2.data[0].cpu().numpy()
    testCategoryNew[sampleId] = yCategoryNew.data[0].cpu().numpy()
    testColor[sampleId] = yColor.data[0].cpu().numpy()
    testGender[sampleId] = yGender.data[0].cpu().numpy()
    testSleeve[sampleId] = ySleeve.data[0].cpu().numpy()

# Save results
result = {"hn2": testHn2}
savemat("encodeHn2Rnn1002Full.mat", result)