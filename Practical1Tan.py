from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Make sure to remove the "ram_crash" file to get a clean run.
# To fix ram error
crashfile = "ram_crash"
def save_object(network, epoch, testdata, validationdata, crashed, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        network_dump = {"network":network, "epoch":epoch, "testdata":testdata, "validationdata":validationdata, "crashed":crashed}
        pickle.dump(network_dump, output, pickle.HIGHEST_PROTOCOL)




batch_size = 200
epochs = 25
learning_rate = 0.0001

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transforms.ToTensor())
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transforms.ToTensor())

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))


train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()

#plt.imshow(np.transpose(torchvision.utils.make_grid(images).numpy(),(1,2,0)))
#plt.show()


def createNetwork():
    return nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(8192, 10),
    nn.Softmax(),
    )

# Checks if anychrash happend
start = 0
next_epoch = 0
network = createNetwork()
ramError = False
networkcopy = copy.deepcopy(network)
optimizer = optim.Adam(network.parameters(), lr = learning_rate)
loss_function = nn.CrossEntropyLoss()
firstRun = True

for epoch in range(start,epochs+next_epoch):
    new_trainingloss = 0
    #try:
    i = 0
    for train_nr, (images, labels) in enumerate(train_loader):
        i += 1
        optimizer.zero_grad()

        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        print(
            '\rEpoch {} [{}/{}] - Loss: {}'.format(
                epoch+1, train_nr+1, len(train_loader), loss
            ),
            end=''
        )
        new_trainingloss += loss.item()
    writer.add_scalar('Tanh/traininglosses', new_trainingloss/i, epoch)
        #traininglosses.append(new_trainingloss)
    new_validationloss = 0
        
    i = 0
    for val_nr, (images, labels) in enumerate(validation_loader):
        i += 1
        prediction = network(images)
        new_validationloss += loss_function(prediction, labels).item()

    if firstRun:
        validationloss = new_validationloss
        firstRun = False
    elif  new_validationloss < validationloss:
        print("\rnew best",end = "")
        validationloss = new_validationloss
        networkcopy = copy.deepcopy(network)

        #validationlosses.append(validationloss.item())
    writer.add_scalar('Tanh/validationloss', validationloss/i, epoch)
    #except:
    #    epochs = epoch
    #    ramError = True
    #    break
corr = 0
'''
if ramError:
    save_object(network, epochs, traininglosses, validationlosses, ramError, crashfile)
    print("\nProblem with ram. Please restart me.")
else:
'''
#save_object(network, epochs, traininglosses, validationlosses, ramError, crashfile)
for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(networkcopy(image), dim=-1)
    corr += (guess == label).sum()
print("\n","Result on test:", 100*corr.item()/10000)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                    {'hparam/accuracy': 100*corr.item()/10000,"hparam/network":3})
    #print(traininglosses)
