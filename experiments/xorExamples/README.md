# Learning XOR for testing

The XOR function is implemented in the following truth table:

|   x   |   y   | XOR(x,y) |
|:-----:|:-----:|:--------:|
| False | False |   False  |
| False |  True |   True   |
| True  | False |   True   |
| True  |  True |   False  |

As discussed in Minsky and Papert (1969), the XOR function is not linearly separable and therefore cannot be learned 
by a linear classifier. It can, however, easily be learned by a non-linear classifier with nine parameters 
(Cybenko, 1989). 

This makes it a nice test case for examining infrastructure. 

# Models

The following code implements MLP as a XOR classifier. 
```
class XORClassifier(nn.Module):
    """ 
    Simple MLP for XOR tests. Roughly follows implementation of 
       https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html

    Includes a nonlinear boolean parameter to flip the Sigmoid activation function between the 
    two linear layers on or off .
    """

    def __init__(self, in_features, num_classes, hidden_features, nonlinear = True):
        super(XORClassifier, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)

        if nonlinear == True:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.linear2 = nn.Linear(hidden_features, num_classes)

        # Overide default initialization with a N(0,1) distribution 
        self.linear1.weight.data.normal_(0, 1)
        self.linear2.weight.data.normal_(0, 1)

        # Log the initialized values for testing reproducibility
        logger.info("Initialized two linear layers with the following weights:")
        logger.info(self.linear1.weight)
        logger.info(self.linear2.weight)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x) 
        x = self.linear2(x)
        return x

class XORModel:
    """
    Juneberry wrapper class for the XOR tests
    """

    def __call__(self, in_features=2, num_classes=2, hidden_features=2, nonlinear = True):

        # Juneberry requires classes to be natural counts, while the pytorch
        # binary loss functions, BCELoss and MSELoss, require a single output node.
        # 
        # Consequently, decrease num_class when it is set to two. 
        if num_classes == 2:
            num_classes = 1

        return XORClassifier(in_features, num_classes, hidden_features, nonlinear)
```

The model can fit XOR with only nine parameters, if configured with
```
    "model_architecture": {
        "module": "juneberry.architectures.pytorch.sample_tabular_nn.XORModel",
        "args": {
            "in_features" : 2, 
            "num_classes": 2,
            "hidden_features": 2,
            "nonlinear": true
        }
    }
```

The model cannot fit XOR, irrespective of the number of parameters, because the two linear layers will reduce to a 
simple linear classifier regardless of the number of additional hidden features. For example, the following 
configuration will never converge:
```
    "model_architecture": {
        "module": "juneberry.architectures.pytorch.sample_tabular_nn.XORModel",
        "args": {
            "in_features" : 2, 
            "num_classes": 2,
            "hidden_features": 2,
            "nonlinear": false
        }
    }
```

# Methods

The XOR function has four training points, which, for our purposes are also the validation data points. 

We have several ways to learn the XOR function across GPUs and batch sizes for a given learning rate using 
SGD. 

TODO: Convert to runs table

* Single GPU
    * Run 0: Batch size = 4, learning rate = 0.1; one mini batch of size 4
    * Run 1: Batch size = 2, learning rate = 0.1; two mini batches of size 2
    * Run 2: Batch size = 2, learning rate = 0.05; two mini batches of size 2
    * Run 3: Batch size = 1, learning rate = 0.1; four mini batches of size 1
    * Run 4: Batch size = 1, learning rate = 0.05; four mini batches of size 1
    * Run 5: Batch size = 1, learning rate = 0.025; four mini batches of size 1
* Two GPUS
    * Run 6: Batch size = 4, learning rate = 0.1; this distributes a minibatch of size 2 to each GPU. 
    * Run 7: Batch size = 2, learning rate = 0.1; this distributes two minibatches of size 1 to each GPU. 
* Four GPUS
    * Run 8: Batch size = 4; learning rate = 0.1; this distributes a minibatch of size 1 to each GPU. 

We expect to find equivalent learning trajectories across the following runs:

* Run 2 = Run 6; Single GPU; Batch size = 2, learning rate = 0.05 should be equivalent to Two GPUS; Batch size = 4, 
* learning rate = 0.10
* Run 4 = Run 7; Single GPU; Batch size = 1, learning rate = 0.05 should be equivalent to Two GPUS; Batch size = 2, 
* learning rate = 0.10
* Run 5 = Run 8; Single GPU; Batch size = 1, learning rate = 0.025 should be equivalent to Four GPUS; Batch size = 4; 
* learning rate = 0.1

We probe this in two ways, first with the linear classifier, which will not converge, and the MLP, which will. 

# Results


# References 

Cybenko, G. Approximation by superposition of a sigmoidal function. Mathematics of Control, Signal and Systems, 
2:303-314, 1989.

Minsky, M., & Papert, S. A. (1969). Perceptrons: An introduction to computational geometry. MIT press.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
