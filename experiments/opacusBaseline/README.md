# Experiment to test opacus on CIFAR-10

## Background

Opacus provides a few examples for DP training of CIFAR-10. 

* [Building an Image Classifier with Differential Privacy](https://opacus.ai/tutorials/building_image_classifier). 

  * [Notebook version](https://github.com/pytorch/opacus/blob/master/tutorials/building_image_classifier.ipynb)

  * Source code: https://github.com/pytorch/opacus/blob/master/examples/cifar10.py
  
  * Simple convnet in 20 epochs, gets $\epsilon = 49.27$, $\delta = 1e-5 < \frac{1}{50,000}$, training accuracy = 63.72, test accuracy = 59.24

  * Resnet-18 with gets 56.61 accuracy with an $\epsilon = 53.54$ 

  * Both use RMSProp with a learning rate of $1e-3$ and a constant schedule. 

  * Tricks: 
     
     * Generally speaking, differentially private training is enough of a regularizer by itself. Adding any more regularization (such as dropouts or data augmentation) is unnecessary and typically hurts performance.

     * Tuning MAX_GRAD_NORM is very important. Start with a low noise multiplier like .1, this should give comparable performance to a non-private model. Then do a grid search for the optimal MAX_GRAD_NORM value. The grid can be in the range [.1, 10].
     
     * You can play around with the level of privacy, EPSILON. Smaller EPSILON means more privacy, more noise -- and hence lower accuracy. Reducing EPSILON to 5.0 reduces the Top 1 Accuracy to around 53%. One useful technique is to pre-train a model on public (non-private) data, before completing the training on the private training data. See the workbook at bit.ly/opacus-dev-day for an example.
  
* [Opacus dev day workbook](bit.ly/opacus-dev-day)

  * This uses a resnet-18 pretrained on ImageNet as a fixed feature extractor for a small, private head trained on CIFAR-10. That's a little weird from a science perspective, but it makes a nice, quick demo. 

  * The ImageNet pre-trained resnet-18 finetuned on CIFAR-10 for 30 epochs gets $\epsilon = 3.10$, $\delta = 1e-5 < \frac{1}{50,000}$, training accuracy = 70.15, test accuracy = 68.50

  * A few things to call out: 

     * Since they are using a model pre-trained on imagenet, they normalize by the imagenet means, not the CIFAR-10 means. 

     * They use SGD with an initial learning rate of $0.1$, a momentum of $0.9$, and Nesterov set to true. The schedule is  proportional steps, hardcoded to be $0.1$ for epochs 1-10, $0.1 / 2 = 0.05$ for epochs 11-20, and  $0.1 / 2 / 3 = 0.01667$ for epochs 21-30. 
     
     * They report an (epsilon, delta) pair for a given (best) alpha. That makes sense from an RDP standpoint. 

     * The have a few warnings that they ignore. The first two are self-explanatory; the last one appears to be pytorch complaining about how opacus futzes with the gradients in order to implement DP. I believe it is related to [this discussion on their GitHub](https://github.com/pytorch/opacus/issues/9). 

       *  `/usr/local/lib/python3.7/dist-packages/opacus/privacy_engine.py:523: UserWarning: A ``sample_rate`` has been provided.Thus, the provided ``batch_size``and ``sample_size`` will be ignored.`

       *  `/usr/local/lib/python3.7/dist-packages/opacus/privacy_engine.py:195: UserWarning: Secure RNG turned off. This is perfectly fine for experimentation as it allows for much faster training performance, but remember to turn it on and retrain one last time before production with ``secure_rng`` turned on.`

       * `/usr/local/lib/python3.7/dist-packages/torch/nn/modules/module.py:795: UserWarning: Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.`
  
## Setup

1. Install juneberry. 
2. Install opacus. 

   1. Note that installing a secure random number generator is necessary for gaurantees to hold. This can be a bit of a pain depending on your version of pytorch and cuda. Info here: https://github.com/pytorch/csprng

3. Run the experiments from the experiment_outline. Two options 

   * Single thread, one GPU
   
   ```
   juneberry$ CUDA_VISIBLE_DEVICES=0 jb_run_experiment -w . -d /datasets/ opacusBaseline -X
   ```

   * Multiple GPUs, one GPU per task

   ```
   juneberry$ jb_run_experiment opacusBaseline
   juneberry$ jb_rules_to_pydoit opacusBaseline main --parallel
   juneberry$ doit -d . -f experiments/opacusBaseline/main_dodo.py -n 6
   ```

Note: If you run into errors, you can more effectively troubleshoot with a single training execution. Example command:

```
CUDA_VISIBLE_DEVICES=0 jb_train -w . -d /datasets/ model_tests/opacus_baselines/resnet20_dpsgd
```

## Methods

This experiment compares two models:

* A CIFAR-10 style resnet 20 from He et al. (2015)
* A preactivation version from He et al. (2016)

Across three different target epsilons:

* 2
* 5
* 10

For three values of grad clipping centered around 1. "We notice the gradient distribution of CNNs on MNIST and CIFAR-10 might be
symmetric and a clipping threshold around 1 works well." [Chin et al., 2020](https://proceedings.neurips.cc/paper/2020/file/9ecff5455677b38d19f49ce658ef0608-Paper.pdf)

* .8333
* 1
* 1.2

We use three different learning rates on a once cycle schedule. The [Opacus docs](https://opacus.ai/docs/faq#my-model-doesnt-converge-with-default-privacy-settings-what-do-i-do) suggest: 

"The next parameter to adjust would be the learning rate. Compared to the non-private training, Opacus-trained models converge with a smaller learning rate (each gradient update is noisier, thus we want to take smaller steps)." 

[Smith and Topin (2018)](https://arxiv.org/pdf/1708.07120.pdf) report non-private results for the one cycle scheduler of 
         
| Dataset  | Architecture | CLR/SS/PL  | CM/SS        | WD   | Epochs | Accuracy (%)", |
|----------|--------------|------------|--------------|------|--------|----------------|
| Cifar-10 | wide resnet  | 0.1-0.5/12 | 0.95-0.85/12 | 10−4 | 25     | 87.3 ± 0.8",   |
| Cifar-10 | wide resnet  | 0.1-1/23   | 0.95-0.85/23 | 10−4 | 50     | 91.3 ± 0.1"    |

Suggesting a reasonable parameter sweep would start target a maximum learning rate of 1.0 and decrease from there. We choose: 

* .01
* .1
* 1

for the two exemplar choices of epochs, 

* 25
* 50


## Results

If you want to follow your results live, you can catch them with the following one-liner: 

```
juneberry$ find models/opacusBaseline -name log_train.txt -printf "grep -H time_sec %p | tail -n 1 \n" | sh | awk '{print $1, $13, $14, $17, $18, $14-$18}' | sort -n -k 5 | sed 's/log_train\.txt\:/output.png /'
```

The experiment runner generates output similar to this, where we have manually added the values from the configs. 


| Arch      | Epsilon |   Grad Clip   | Max LR | Epochs |  Training Accuracy | Validation Accuracy  |       Test Accuracy      |
|-----------|:-------:|:-------------:|:------:|:------:|:------:|:-----------:|:---------------:|
| resnet-18 |    2    |        0.8333 |  0.01  |   25   | 0.2858 |    0.2836   |         0.2903  |
| resnet-18 |    2    |        0.8333 |  0.01  |   50   | 0.3732 |    0.368    |         0.3715  |
| resnet-18 |    2    |        0.8333 |   0.1  |   25   | 0.4275 |    0.4154   |         0.4183  |
| resnet-18 |    2    |        0.8333 |   0.1  |   50   |  0.473 |    0.459    |         0.4495  |
| resnet-18 |    2    |        0.8333 |    1   |   25   | 0.3934 |    0.3791   |         0.3846  |
| resnet-18 |    2    |        0.8333 |    1   |   50   | 0.4068 |    0.4047   |         0.4062  |
| resnet-18 |    2    |       1       |  0.01  |   25   | 0.3224 |    0.317    |         0.3265  |
| resnet-18 |    2    |       1       |  0.01  |   50   | 0.3803 |    0.3762   |         0.3586  |
| resnet-18 |    2    |       1       |   0.1  |   25   | 0.4064 |    0.3975   |         0.4030  |
| resnet-18 |    2    |       1       |   0.1  |   50   | 0.4914 |    0.4848   |         0.4817  |
| resnet-18 |    2    |       1       |    1   |   25   | 0.4154 |    0.4141   |         0.4102  |
| resnet-18 |    2    |       1       |    1   |   50   | 0.3965 |    0.392    |         0.3922  |
| resnet-18 |    2    |      1.2      |  0.01  |   25   | 0.3024 |    0.2924   |         0.3030  |
| resnet-18 |    2    |      1.2      |  0.01  |   50   | 0.4001 |    0.3939   |         0.4006  |
| resnet-18 |    2    |      1.2      |   0.1  |   25   | 0.4296 |    0.4311   |         0.4273  |
| resnet-18 |    2    |      1.2      |   0.1  |   50   | 0.4648 |    0.4521   |         0.4525  |
| resnet-18 |    2    |      1.2      |    1   |   25   | 0.4051 |    0.3951   |         0.4061  |
| resnet-18 |    2    |      1.2      |    1   |   50   | 0.4133 |    0.4041   |         0.4050  |
| resnet-18 |    5    |        0.8333 |  0.01  |   25   | 0.2869 |    0.2812   |         0.2915  |
| resnet-18 |    5    |        0.8333 |  0.01  |   50   | 0.3793 |    0.3693   |         0.3800  |
| resnet-18 |    5    |        0.8333 |   0.1  |   25   |  0.471 |    0.4584   |         0.4285  |
| resnet-18 |    5    |        0.8333 |   0.1  |   50   | 0.5063 |    0.4928   |         0.4643  |
| resnet-18 |    5    |        0.8333 |    1   |   25   | 0.4573 |    0.4469   |         0.4502  |
| resnet-18 |    5    |        0.8333 |    1   |   50   | 0.4966 |    0.4746   |         0.4848  |
| resnet-18 |    5    |       1       |  0.01  |   25   | 0.3386 |    0.3275   |         0.3400  |
| resnet-18 |    5    |       1       |  0.01  |   50   | 0.4153 |    0.4037   |         0.4162  |
| resnet-18 |    5    |       1       |   0.1  |   25   | 0.4891 |    0.4723   |         0.4420  |
| resnet-18 |    5    |       1       |   0.1  |   50   | 0.5389 |    0.5145   |         0.5092  |
| resnet-18 |    5    |       1       |    1   |   25   | 0.4585 |    0.4379   |         0.4515  |
| resnet-18 |    5    |       1       |    1   |   50   | 0.4773 |    0.4619   |         0.4795  |
| resnet-18 |    5    |      1.2      |  0.01  |   25   | 0.3273 |    0.3199   |         0.3284  |
| resnet-18 |    5    |      1.2      |  0.01  |   50   |  0.425 |    0.4037   |         0.3642  |
| resnet-18 |    5    |      1.2      |   0.1  |   25   | 0.5011 |    0.4883   |         0.4915  |
| resnet-18 |    5    |      1.2      |   0.1  |   50   | 0.5391 |    0.5129   |         0.5072  |
| resnet-18 |    5    |      1.2      |    1   |   25   | 0.4818 |    0.4721   |         0.4788  |
| resnet-18 |    5    |      1.2      |    1   |   50   | 0.4789 |    0.4717   |         0.4709  |
| resnet-18 |    10   |        0.8333 |  0.01  |   25   | 0.3113 |    0.3035   |         0.3193  |
| resnet-18 |    10   |        0.8333 |  0.01  |   50   |  0.384 |    0.3701   |         0.3818  |
| resnet-18 |    10   |        0.8333 |   0.1  |   25   | 0.4927 |    0.4785   |         0.4264  |
| resnet-18 |    10   |        0.8333 |   0.1  |   50   | 0.5585 |    0.5377   |         0.5124  |
| resnet-18 |    10   |        0.8333 |    1   |   25   | 0.5197 |    0.5098   |         0.5023  |
| resnet-18 |    10   |        0.8333 |    1   |   50   | 0.5169 |    0.5111   |         0.5104  |
| resnet-18 |    10   |       1       |  0.01  |   25   | 0.3454 |    0.3391   |         0.3516  |
| resnet-18 |    10   |       1       |  0.01  |   50   | 0.4134 |    0.4008   |         0.4028  |
| resnet-18 |    10   |       1       |   0.1  |   25   | 0.5231 |    0.5105   |         0.5098  |
| resnet-18 |    10   |       1       |   0.1  |   50   |  0.557 |    0.5426   |         0.5237  |
| resnet-18 |    10   |       1       |    1   |   25   | 0.5204 |    0.5047   |         0.5016  |
| resnet-18 |    10   |       1       |    1   |   50   | 0.5298 |    0.5193   |         0.5142  |
| resnet-18 |    10   |      1.2      |  0.01  |   25   |  0.351 |    0.3416   |         0.3545  |
| resnet-18 |    10   |      1.2      |  0.01  |   50   | 0.4372 |    0.4334   |         0.4227  |
| resnet-18 |    10   |      1.2      |   0.1  |   25   | 0.5248 |    0.5154   |         0.5103  |
| resnet-18 |    10   |      1.2      |   0.1  |   50   | 0.5529 |    0.5309   |         0.5220  |
| resnet-18 |    10   |      1.2      |    1   |   25   | 0.5438 |    0.5275   |         0.5354  |
| resnet-18 |    10   |      1.2      |    1   |   50   | 0.5484 |    0.5414   |         0.5391  |
| preact-18 |    2    |        0.8333 |  0.01  |   25   | 0.2826 |    0.2768   |         0.2774  |
| preact-18 |    2    |        0.8333 |  0.01  |   50   | 0.3797 |    0.3746   |         0.3755  |
| preact-18 |    2    |        0.8333 |   0.1  |   25   | 0.3888 |    0.3756   |         0.3807  |
| preact-18 |    2    |        0.8333 |   0.1  |   50   | 0.4503 |    0.4375   |         0.4230  |
| preact-18 |    2    |        0.8333 |    1   |   25   |  0.423 |    0.4115   |         0.4210  |
| preact-18 |    2    |        0.8333 |    1   |   50   | 0.4298 |    0.4266   |         0.4345  |
| preact-18 |    2    |       1       |  0.01  |   25   | 0.2866 |    0.2732   |         0.2869  |
| preact-18 |    2    |       1       |  0.01  |   50   | 0.3716 |    0.3572   |         0.3429  |
| preact-18 |    2    |       1       |   0.1  |   25   | 0.3832 |    0.3732   |         0.3760  |
| preact-18 |    2    |       1       |   0.1  |   50   |  0.456 |    0.4445   |         0.4403  |
| preact-18 |    2    |       1       |    1   |   25   | 0.4028 |    0.3916   |         0.4021  |
| preact-18 |    2    |       1       |    1   |   50   | 0.3992 |    0.3967   |         0.4034  |
| preact-18 |    2    |      1.2      |  0.01  |   25   |  0.342 |    0.3346   |         0.3404  |
| preact-18 |    2    |      1.2      |  0.01  |   50   | 0.4026 |    0.382    |         0.3922  |
| preact-18 |    2    |      1.2      |   0.1  |   25   | 0.4039 |    0.3953   |         0.3975  |
| preact-18 |    2    |      1.2      |   0.1  |   50   | 0.4374 |    0.4301   |         0.4191  |
| preact-18 |    2    |      1.2      |    1   |   25   | 0.4057 |    0.3945   |         0.4090  |
| preact-18 |    2    |      1.2      |    1   |   50   | 0.4041 |    0.4033   |         0.4040  |
| preact-18 |    5    |        0.8333 |  0.01  |   25   | 0.3033 |    0.2953   |         0.2953  |
| preact-18 |    5    |        0.8333 |  0.01  |   50   | 0.3814 |    0.3643   |         0.3732  |
| preact-18 |    5    |        0.8333 |   0.1  |   25   | 0.4703 |    0.4436   |         0.4429  |
| preact-18 |    5    |        0.8333 |   0.1  |   50   | 0.5433 |    0.5311   |         0.4940  |
| preact-18 |    5    |        0.8333 |    1   |   25   | 0.4765 |    0.4635   |         0.4814  |
| preact-18 |    5    |        0.8333 |    1   |   50   | 0.4897 |    0.4762   |         0.4887  |
| preact-18 |    5    |       1       |  0.01  |   25   | 0.3335 |    0.3326   |         0.3332  |
| preact-18 |    5    |       1       |  0.01  |   50   | 0.3983 |    0.3836   |         0.3891  |
| preact-18 |    5    |       1       |   0.1  |   25   | 0.4614 |    0.4398   |         0.4500  |
| preact-18 |    5    |       1       |   0.1  |   50   | 0.5437 |    0.5248   |         0.5209  |
| preact-18 |    5    |       1       |    1   |   25   | 0.4753 |    0.4568   |         0.4723  |
| preact-18 |    5    |       1       |    1   |   50   | 0.4825 |    0.4664   |         0.4641  |
| preact-18 |    5    |      1.2      |  0.01  |   25   | 0.3463 |    0.3377   |         0.3497  |
| preact-18 |    5    |      1.2      |  0.01  |   50   | 0.4124 |    0.4012   |         0.4016  |
| preact-18 |    5    |      1.2      |   0.1  |   25   | 0.4814 |     0.46    |         0.4603  |
| preact-18 |    5    |      1.2      |   0.1  |   50   | 0.5249 |    0.5037   |         0.4985  |
| preact-18 |    5    |      1.2      |    1   |   25   | 0.4801 |    0.4637   |         0.4642  |
| preact-18 |    5    |      1.2      |    1   |   50   | 0.4988 |    0.4918   |         0.4902  |
| preact-18 |    10   |        0.8333 |  0.01  |   25   | 0.2963 |    0.2869   |         0.2921  |
| preact-18 |    10   |        0.8333 |  0.01  |   50   | 0.3848 |    0.3754   |         0.3741  |
| preact-18 |    10   |        0.8333 |   0.1  |   25   | 0.4985 |    0.4826   |         0.4386  |
| preact-18 |    10   |        0.8333 |   0.1  |   50   | 0.5692 |    0.5469   |         0.5071  |
| preact-18 |    10   |        0.8333 |    1   |   25   | 0.5306 |    0.5293   |         0.5213  |
| preact-18 |    10   |        0.8333 |    1   |   50   |  0.53  |    0.5111   |         0.5190  |
| preact-18 |    10   |       1       |  0.01  |   25   | 0.3212 |    0.2996   |         0.3206  |
| preact-18 |    10   |       1       |  0.01  |   50   | 0.4087 |    0.4043   |         0.4083  |
| preact-18 |    10   |       1       |   0.1  |   25   | 0.5313 |    0.5105   |         0.4809  |
| preact-18 |    10   |       1       |   0.1  |   50   | 0.5761 |    0.5584   |         0.5439  |
| preact-18 |    10   |       1       |    1   |   25   | 0.5025 |    0.4871   |         0.4718  |
| preact-18 |    10   |       1       |    1   |   50   | 0.5302 |    0.509    |         0.5208  |
| preact-18 |    10   |      1.2      |  0.01  |   25   | 0.3554 |    0.3475   |         0.3607  |
| preact-18 |    10   |      1.2      |  0.01  |   50   | 0.4277 |    0.4131   |         0.3756  |
| preact-18 |    10   |      1.2      |   0.1  |   25   | 0.5154 |    0.5057   |         0.4877  |
| preact-18 |    10   |      1.2      |   0.1  |   50   | 0.5681 |    0.5494   |         0.5290  |
| preact-18 |    10   |      1.2      |    1   |   25   | 0.5354 |    0.5219   |         0.5228  |
| preact-18 |    10   |      1.2      |    1   |   50   | 0.5539 |    0.5436   |         0.5458  |


## Discussion 

The best runs by epsilon are: 

* Epsilon = 2: resnet-18, grad clip = 1, max_lr = 0.1, epochs = 50, Test Accuracy: 48.17%
* Epsilon = 5: preact-18, grad clip = 1, max_lr = 0.1, epochs = 50, Test Accuracy: 52.09%
* Epsilon = 10: preact-18, grad clip = 1.2, max_lr = 1, epochs = 50, Test Accuracy: 54.58%

This is not particularly close to benchmark results reported in [Papernot et al. (2020)](https://arxiv.org/abs/2007.14191):

    "Finally, on CIFAR10, we achieve 66.2% test accuracy at (ε, δ) = (7.53, 10−5) in a setup for which prior work achieved 61.6%."

But, our results were based on simplying applying Opacus to off the shelf non-private components. Future work could include:

* Implementing tanh activations instead of relu. [Papernot et al. (2020)](https://arxiv.org/abs/2007.14191)
* Setting the weight decay to zero. [Tramer and Boneh (2021)](https://arxiv.org/pdf/2011.11660.pdf)
* Using a secure random number generator for publication quality results. 

On the last point, that can be achieved by modifying the privacy_engine stanza to include `"secure_rng": true`: 

```
"privacy_engine": { 
    ...
    "secure_rng": true },
```

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
