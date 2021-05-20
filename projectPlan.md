# The project plan for ADL Project - Pneumonia Detection with Chest X-RayImages

## Architechture

The prepossessing we have discussed is for now just padding.

For this project we will use a CNN.

The first network we will try is:

The Convolution part will be VGG3 with pooling. The activation layer 
will be LeakyReLU.

This will then be feeded in to a 2 layer classification network that gives 
2 outputs. We are considering to have a Softmax in the end.

The loss function we will use is the Binary cross entropy function.

Optimizer will be Adam.

If needed we will start with using following regularization technics:
Dropout and random crop.

If we get good result fast we will try having 3 outputs and see if we can 
classify if it is bacteria or virus that have caused the infection.

Or we might make a new CNN for that task. 

We will also create confusion matrices so that we can see how our network 
behaves.



## Dataset

We have setup a few rules for the dataset. 
1. Change so we have 2 validationsets. 1 we use during training and 1 we use as a internal testset.
2. Only use the real testset one time.
