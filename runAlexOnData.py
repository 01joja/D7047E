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
epochs = 2
learning_rate = 0.0001

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=preprocess)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=preprocess)

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))


train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

dataiter = iter(train_loader)
images, labels = dataiter.next()

#plt.imshow(np.transpose(torchvision.utils.make_grid(images).numpy(),(1,2,0)))
#plt.show()



# Import alexnet
alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=True)
alex.eval()

# Stop learning in the original network
#for param in alex.parameters():
#    param.requires_grad=False
# Creates a new hidden layer
alex.classifier[6]=nn.Linear(4096,10)
print(alex)


optimizer = optim.Adam(alex.parameters(), lr = learning_rate)
loss_function = nn.CrossEntropyLoss()

for epoch in range(epochs):
    new_trainingloss = 0
    i = 0
    for train_nr, (images, labels) in enumerate(train_loader):
        i += 1
        optimizer.zero_grad()

        prediction = alex(images)
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
    writer.add_scalar('Adam/traininglosses', new_trainingloss/i, epoch)
    new_validationloss = 0
        
    i = 0
    for val_nr, (images, labels) in enumerate(validation_loader):
        i += 1
        prediction = alex(images)
        new_validationloss += loss_function(prediction, labels).item()
    writer.add_scalar('Adam/validationloss', new_validationloss/i, epoch)


corr = 0
i = 0
for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(alex(image), dim=-1)
    result = (guess == label).sum()
    corr += result.item()
    if 0 == (i%30):
        print("\r", "Right guess:", 100*corr/i, "Tested pictures:", 100*i/10000,end="")
print("\n","Result on test:", 100*corr/10000)
writer.add_hparams({'lr': learning_rate, 'bsize': batch_size},
                    {'hparam/accuracy': 100*corr/10000})
                    