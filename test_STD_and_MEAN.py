# Insperation is from https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/5
# Will do this after I have loaded the dataset



mean = 0.
std = 0.
nb_samples = 0.
for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples