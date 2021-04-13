import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import pickle


# Function that can save files
def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
        
batch_size = 1000
epochs = 20

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar_data = datasets.CIFAR10("./", train=True, download=True, transform=transform)
cifar_test = datasets.CIFAR10("./", train=False, download=True, transform=transform)

cifar_train, cifar_val = torch.utils.data.random_split(cifar_data,[40000,10000],generator=torch.Generator().manual_seed(420))

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)


network = nn.Sequential(
    
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1),
    nn.Tanh(),
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,padding=1),
    nn.Tanh(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(8192, 10)
)

print(network)

val_images, val_labels = next(iter(validation_loader))
optimizer = optim.SGD(network.parameters(), lr = 0.0001)
loss_function = nn.CrossEntropyLoss()
traininglosses = []
validationlosses = []
firstRun = True
validationloss = 0

for epoch in range(epochs):
    for train_nr, (images, labels) in enumerate(train_loader):
        prediction = network(images)
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (train_nr+1)%10 == 0:
            print(
                '\rEpoch {} [{}/{}] - Loss: {}'.format(
                    epoch+1, train_nr+1, len(train_loader), loss
                ),
                end=''
            )
        traininglosses.append(loss.item())
        
    prediction = network(val_images)
    new_validationloss = loss_function(prediction, val_labels)

    if firstRun:
        validationloss = new_validationloss
        firstRun = False
    elif  new_validationloss.item() < validationloss:
        print("\n New best:", new_validationloss.item())
        validationloss = new_validationloss.item()
        networkcopy = copy.deepcopy(network)
        save_object(networkcopy,"best_network")

    validationlosses.append(validationloss)

train_loader = DataLoader(cifar_train, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(cifar_val, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(cifar_test, batch_size= batch_size, shuffle=False)

corr = 0

for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(networkcopy(image), dim=-1)
    corr += (guess == label).sum()
print("\n","Best network result on test:", corr/10000)
corr = 0
for index, (image, label) in enumerate(test_loader):
    guess = torch.argmax(networkcopy(image), dim=-1)
    corr += (guess == label).sum()
print("\n","Result lastnetwork on test:", corr/10000)

