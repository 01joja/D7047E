# Insperation is from https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5
# Will do this after I have loaded the dataset

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import load_dataset

preprocessTraining = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize([2500,2500]),
    transforms.ToTensor(),
])

path = load_dataset.getTrainPath()
Dataset =load_dataset.PneumoniaDataSet(path, transform = preprocessTraining)
loader = DataLoader(
    Dataset,
    batch_size=10,
    #num_workers=2,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data[0].size(0)
    data = data[0].view(batch_samples, data[0].size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples
    print(
        '\rLooked at {} %, of the pictures'.format(
            (100*nb_samples/5232)
        ),
        end='                                                 '
    )

mean /= nb_samples
std /= nb_samples

print(mean)
print(std)

f = open("STD_and_Mean.txt", "w")
f.write("Calculated STD: "+str(std)+" Mean: "+ str(mean))
f.close()