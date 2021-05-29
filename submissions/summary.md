# A summary of all the lectures

## CNN:

Usages:
1. Images
2. [Speech recognition](https://ieeexplore.ieee.org/document/6857341?reload=true)
3. [Text Classification](https://arxiv.org/abs/1509.01626)
4. and more

Uses kernels to go over the data (lets say images) to find features that can be given to an ANN. If you have more then one kernel you will get more outputs(features).

[Backpropergation](https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c) It's the first YouTube link but in text.

Regularization technics:
Random cropping
dropout
Introduce noise

Other things:
Transfer learning or fine tuning.
Max-pooling.

CNN architectures: 
- AlexNet
  - 224x224x3 input 3 convolutions 11x11, 5x5, 3x3 all with max-pool(3x3). ANN 4096->4096->1000
- VGG-16 (16 layers with waights deep)
  - input 224x224x3 2 conv(3x3) max-pool(2x2) 3 conv(3x3) max-pool(2x2). ANN 4096->4096->1000
- VGG-16 (19 layers deep)
- ResNet (152 layers and introduction of residual blocks)
- Squeeze Net (A small network)
- Inception Module
  - Many different convolutions(1x1,3x3,5x5,max-pool) in one layer. It is used to make it less computational hungry. This makes it wider and not deeper. I think Marcus mentioned that this will keep more of the original data for the next layer.
- Inception Residual unit
  - Uses both Inception and residual block in one.
- PolyNet
  - It uses poly-2, mpoly-2 and 2-way. To make more diverse decisions. This got better result then Inception-ResNet on the same computation budget.
- Densely Connected Convolution Network
  - Have sections of convolution x layers. After one section it passes the information froward to the other sections later in the network.

Activation functions:
- ReLU (The golden standard)
  - everything below zero is zero. Rest is unchanged.
- Leaky ReLU
  - everything below zero is equal to 0.1 etcetera.
- Sigmoid
  - e^x/(e^x+1)
  - Squashes all inputs to something between 0 and 1.
- tanh
  - e^x-e^-x/(e^x+e^-x)
  - Squashes all inputs to something between -1 and 1.

Filters:
First more small features like lines and edges.
Then it will get more and more specific the more and more layers you have.

## Visualization:

Sparsity
- ?  

CNNVis
- Looks for what neurons that get activated in an activation function over a lot of different images. Then you can follow and see how much that will impact the next layer and what neuron will get activated. With this you can see if the network is too wide (similar looking activation matrices) and how much one activation matrix impacts the next activation function.

Deconvolution (or transposed convolution?) 
- Problem Max Pooling.
- Uses deconvolution to visualize all features in a CNN. First it puts a image in to the CNN and stores all the values created in the CNN. Then it runs the deconvolution on the values and gets what it fired on. This can help you find daed filters and blocking artifacts. 

Deep Vis:
- Feed a CNN with many pictures of one of the labels it where trained on. Backpropogate for every picture and use that gradient to update a random picture with the result. This will visualize what features in the picture the CNN uses. This can be used to test if a network can be fooled and see how good it can differ on different labels or if they are similar. 

What should an expert look at?
- Higher layers should detect more abstract features
- Investigate similar classes and check purity
- Layers with only positive weights (redundancy)
- Check for redundancy (same neurons, similar act.)
- Check weight-updates

Sammon's mapping:
- A old NLDR techniques

Nonlinear dimensionality reduction
- An ANN.

Kernel PCA (Kernel principal component analysis):
- 

Isomap:
- 

PCA:
- Read more

T-SNE t-distributed stochastic neighbor embedding:
- Lowers the dimensions by trying to keep close points close and faraway points faraway.
  - This is done for every point compered to every other point so it 
- Has two hyperparameters, "preplexity" and "epsilon" and they do make a big difference.
  - Epsilon is learning rate.
  - Perplexity. It is a guess of how many close neighbors a point has. It need to change according to the number of data points. 
- Distance between plots might not mean anything.
- Random noise doesn't always look random.
- It creates a new plot every time.

## GANs

Created with game theory. One network tries to classify pictures in to real and fake. Another network creates images from random noise and tries to fool the classifier. This goes on until the classifier has to guess and can't determine what is fake and not. 

Examples of different GANs:
- Image-to-Image Translation: GauGan
  - You draw a simple image and it can transform it in to a real image.
- Super-Resolution: SRGAN
  - Creates a image with higher pixel rate then the original picture.
- Image-to-Image Translation: CycleGan
  - Changes zebras to horses in a movie or a summer landscape to a winter landscape.
- Text-to-Image: StackGAN
  - Creates high definition images from text.
- Music Generation: MuseGAN
  - Creates music.

Use cases:
- Data Augmentation.
- Privacy:
  - Create similar pictures of for example sick people that have the illness but dose not exist in the real world.
- Face anonymization.
  - Useful for sexual assault victims, witnesses, whistle blowers, activist

Problems:
- Deep fakes:
  - Putting words in people's mouths
  - Identity theft

## RNN, LSTM and GRU

RNN (Recurrent neural network) are used for variable length sequences of input. It's previous values from the hidden layers to evaluate the next input. So the previous input effect the next input. But it has a big problem with vanishing and exploding gradient. This is because its output will be feeded back in to the same hidden layer. So if the weight is any different from 1 the value will explode or vanish over time. Exploding gradient can be solved by a cut off at a specific value but this dose feel like a barbaric why to solve it.

LSTM (long short-term memory) is a way to solve (or at least lessen) the problem with the vanishing or exploding gradient. LSTM consist of a cell state that can be tapped into, changed or forgotten over time. 
Architecture LSTM:
- Dose the forget gate only use 1 or 0?

Different RNNs:
- one to one
- one to many
- many to one 
- and so on.

Use cases:
- Words.
- Books.
- Movies.
- Any data that has a "bigger picture".


## NLP (Natural Language Processing)


