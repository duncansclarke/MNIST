'''
Model 2B: Implementation of a neural network model with predefined library (i.e. PyTorch) for MNIST
Uses different parameters from Model 1
'''

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim

# Define transformation to Torch sensor
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Download datasets
trainset = datasets.MNIST('PATH_TO_STORE_TRAINSET',
                          download=True, train=True, transform=transform)
testset = datasets.MNIST('PATH_TO_STORE_TESTSET',
                         download=True, train=False, transform=transform)
# Load datasets
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)


dataiterator = iter(trainloader)
images, labels = dataiterator.next()

''' Build neural network '''

# Instantiate layer sizes
input_size = 784  # 28x28 pixels = 784 pixels when flattened
hidden_sizes = [128]  # A single hidden layer with 64 neurons
output_size = 10  # 10 outputs - 1 for each number 0-9

# Wrap layers in network with Sigmoidal activation
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.Sigmoid(),
                      nn.Linear(hidden_sizes[0], output_size),
                      nn.LogSoftmax(dim=1))

# Instantiate negative log likelihood loss
criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1)

logps = model(images)
loss = criterion(logps, labels)

# Adjust the weights
loss.backward()

optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
epochs = 15
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Convert mnist images to vector of size 784
        images = images.view(images.shape[0], -1)

        # Training
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # Backpropagating
        loss.backward()

        # Optimizing weights
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e,
                                                    running_loss/len(trainloader)))

''' Test Model '''
# Initialize variables to keep track of accuracy
correct, total = 0, 0
# Initialize empty confusion matrix
c_m = np.zeros(shape=(10, 10)).astype(int)
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 784)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        prob = list(ps.numpy()[0])
        pred = prob.index(max(prob))
        y = labels.numpy()[i]
        # Check if prediction is correct, and increment counter accordingly
        if(y == pred):
            correct += 1
        total += 1  # Update total count
        c_m[pred][y] += 1  # Update confusion matrix

print("\nAccuracy =", (correct/total))
print(c_m)
