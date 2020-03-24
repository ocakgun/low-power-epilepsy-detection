import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
from matplotlib import pyplot as plt

import pickle

# from scipy import ndimage
# from skimage.measure import block_reduce
# torch.set_printoptions(precision=10)
# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_dtype(torch.float64)
# ((outputs[1] == outputs[1].max()).nonzero()) == labels[1]
# model.fc1.weight = torch.nn.Parameter(torch.mul(model.fc1.weight, 2))

# torch.cuda.device(0)

dsWidth = 1  # Downsampling window width
imWidth = 28  # image width/length
imDsWidth = int(imWidth / dsWidth)  # Downsampled image width

# for imDsWidth in np.arange(8,10+1,1):
# print(type(imDsWidth))

# imDsWidth = 5

results = []
# for imDsWidth in np.arange(1,imWidth+1,1):
for imDsWidth in range(1, imWidth + 1, 1):
    input_size, output_size, num_epochs, batch_size = imDsWidth * imDsWidth, 10, 5, 100

    train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize(imDsWidth),
                                                   transforms.ToTensor()
                                                   # transforms.Normalize([0.5], [-0.5])
                                               ]))

    test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, download=True,
                                              transform=transforms.Compose([
                                                  transforms.Resize(imDsWidth),
                                                  transforms.ToTensor()
                                                  # transforms.Normalize([0.5], [-0.5])
                                              ]))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False)


    class SingleLayerNN(nn.Module):
        def __init__(self, input_size, output_size):  # , num_classes):
            super(SingleLayerNN, self).__init__()
            self.fc1 = nn.Linear(input_size, output_size, bias=False)
            # self.softmin = nn.Softmax() #ReLU()
            # self.softmin = nn.ReLU() #ReLU()
            # self.softmin = nn.Softmax(dim=0) #ReLU(
            self.relu = nn.ReLU()
            self.softmin = nn.Softmin(dim=-1)  # ReLU(

        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.softmin(out)  # relu(out)
            return out


    # class WeightClipper(object):
    # def __init__(self, frequency=1):
    # self.frequency = frequency

    # def __call__(self, module):
    # if hasattr(module, 'weight'):
    # w = module.weight.data
    # w = torch.clamp(w, min = 0, max= 15)

    # clipper = WeightClipper()

    model = SingleLayerNN(input_size, output_size)  # , num_classes)
    # model.apply(clipper)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.RMSprop(model.parameters(), lr = 0.01)#, lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, imDsWidth * imDsWidth)
            labels = labels

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # model.fc1.weight.data = model.fc1.weight.data.clamp(0, 15)
            model.fc1.weight.data = model.fc1.weight.data.clamp(min=0)
            # model.fc1.weight.data = model.fc1.weight.data.clamp(min=0,max=1)
            # model.fc1.weight.data = torch.div(torch.ones(model.fc1.weight.shape), model.fc1.weight.data)
            # model.fc1.weight.data = torch.div(model.fc1.weight.data,torch.max(torch.abs(model.fc1.weight.data)))
            # model.fc1.weight.data = model.fc1.weight.data.clamp(min=0)

            # if epoch % clipper.frequency == 0:
            # model.apply(clipper)

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                print(model.fc1.weight.min())
                print(model.fc1.weight.max())

    # modelWeightsFloat = model.fc1.weight.data
    # model.fc1.weight.data = modelWeightsFloat

    weightsFloat = model.fc1.weight.data
    wMax1 = torch.div(weightsFloat, torch.max(weightsFloat))

    accuracyBits = []
    maxElements = []
    for nBits in np.arange(1, 10, 1):
        ### Fix the max to 1
        # nBits = 3
        # working example
        bins = np.arange(0, 1, 1 / (2 ** nBits))
        inds = np.digitize(wMax1.data.numpy(), bins)
        qWeights = bins[inds - 1]
        #### 2nd try
        # bins=[]
        # for i in np.arange(1,2**nBits,1):
        # bins.append(1/(2*i))
        # bins.append(0)
        # bins=np.array(bins)
        # # inds=np.digitize(weightsFloat.data.numpy(), bins)
        # inds=np.digitize(wMax1.data.numpy(), bins)
        # qWeights = bins[inds]
        #
        #
        #
        #
        model.fc1.weight.data = torch.FloatTensor(qWeights)
        # print(torch.max(model.fc1.weight.data))
        # model.fc1.weight.data = modelWeightsFloat.clamp(0, 32)
        # model.fc1.weight.data = torch.add(modelWeightsFloat,torch.abs(torch.min(modelWeightsFloat)))
        # model.fc1.weight.data = torch.div(torch.ones(model.fc1.weight.data.shape), model.fc1.weight.data)
        # model.fc1.weight.data = torch.div(model.fc1.weight.data,torch.max(model.fc1.weight.data))
        # model.fc1.weight.data.clamp(0,1)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                imagesShaped = images.reshape(-1, imDsWidth * imDsWidth)
                labels = labels
                outputs = model(imagesShaped)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            # print('Accuracy of the network on the test images: {} %'.format(100 * correct / total))
            accuracyBits.append(100 * correct / total)
        modelQWeights = model.fc1.weight.data
        maxWeight = 0
        for nCount in np.arange(0, 10, 1):
            # print("Neuron ", nCount, modelQWeights[nCount].nonzero().size(0))
            if (maxWeight < modelQWeights[nCount].nonzero().size(0)):
                maxWeight = modelQWeights[nCount].nonzero().size(0)
        maxElements.append(maxWeight)

    model.fc1.weight.data = weightsFloat

    results.append((imDsWidth, accuracyBits, maxElements))

with open('data1.pickle', 'wb') as f:
    pickle.dump(results, f)

# model.fc1.weight.data = modelWeightsFloat

'''
function = nn.Softmin() 
input = model.fc1.forward(imagesShaped[10])
output = function(input)
outputs[10] == output
'''

# imgNo=2
# if (outputs[imgNo]==outputs[imgNo].max()).nonzero()==labels[imgNo]:
# print("T\tImage: %d\tdigit:%d" % (imgNo, labels[imgNo].tolist()))
# else:
# print("F\tImage: %d\tdigit:%d" % (imgNo, labels[imgNo].tolist()))
# print("\nLabel: ", labels[imgNo])
# print("\n\nInput to the activation: ", model.fc1.forward(imagesShaped[imgNo]))


# print("\n\nOutputs: ", model(imagesShaped[imgNo]))
# # print("\n\nImage data:\n", imagesShaped[imgNo])
# # print("\n\nWeights for the classifier neuron:\n", model.fc1.weight[labels[imgNo]])

# plt.gray()
# plt.imshow(images[imgNo].data.numpy()[0])
# plt.show()

# plt.imshow(modelQWeights[0].reshape(-1,imDsWidth,imDsWidth).data.numpy()[0]);plt.show()

# plt.imshow(modelQWeights[2].reshape(-1,imDsWidth,imDsWidth).data.numpy()[0]);plt.show()

# if (outputs[imgNo]==outputs[imgNo].max()).nonzero()==labels[imgNo]:
# print("T\tImage: %d\tdigit:%d" % (imgNo, labels[imgNo].tolist()))
# else:
# print("F\tImage: %d\tdigit:%d" % (imgNo, labels[imgNo].tolist()))
# print("\nLabel: ", labels[imgNo])
# print("\n\nOutputs: ", model(imagesShaped[imgNo]))
# for iN in np.arange(0,10):
# print(torch.dot(model.fc1.weight.data[iN],imagesShaped[imgNo]))

# mFunc=nn.Softmin()
# for imCount in np.arange(0,100):
# if (outputs[imCount]==outputs[imCount].max()).nonzero()==labels[imCount]:
# print("T\tImage: %d\tdigit:%d\tprobability:%f" % (imCount, labels[imCount].tolist(), torch.max(mFunc(model.fc1.forward(imagesShaped[imCount])))))
# else:
# print("F\tImage: %d\tdigit:%d\tprobability:%f" % (imCount, labels[imCount].tolist(), torch.max(mFunc(model.fc1.forward(imagesShaped[imCount])))))

# print("Nonzero weights after quantization:")
# maxWeight=0
# for nCount in np.arange(0,10):
# print("Neuron ", nCount, modelQWeights[nCount].nonzero().size(0))
# if(maxWeight < modelQWeights[nCount].nonzero().size(0)):
# maxWeight = modelQWeights[nCount].nonzero().size(0)
# print("NN elements per row: ", maxWeight)