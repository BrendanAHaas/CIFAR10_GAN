# CIFAR10_GAN
GAN (Generative Adversarial Network) designed for creating new images, based on training images in the CIFAR-10 dataset (available at https://www.cs.toronto.edu/~kriz/cifar.html).  The results were used to win Peraton's 3rd EDS Machine Learning Challenge.


Introduction

The CIFAR-10 dataset has 60,000 32x32 color images in 10 classes, with 6,000 images per class.  Those 10 classes include airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.  


What is a GAN?

A GAN is, simply, two machine learning models competing against one another.  One model, the Discriminator, tries to guess if an image is real or fake.  The second model, the Generator, tries to produce fake images that can fool the Discriminator.  The Discriminator will take as an input a 32x32x3 matrix (color image), and output whether it is real or fake (typically 1 or 0).  The Generator will take as an input a randomly generated seed or image (size can vary), and output a 32x32x3 matrix (color image).


Primary Issues with GANS
1.  Training Time
When working with images, neural networks tend to work well for modeling, but these can easily reach millions of parameters even for relatively small networks.  As a result, training can take significantly longer, particularly in this challenge (where cloud resources were not utilized).
2.  Stability
If one of the two competing models in the GAN starts significantly outperforming the other, learning will stop and additional training won't help with improving the model.  Typically, the Discriminator learns more quickly (since it has an easier job) and has to be slowed down to prevent it from outcompeting the Generator.
3.  Mode Collapse
The generator may start producing only one or a couple type of images if it finds those images consistently fool the discriminator.  Since the goal of the project is to be able to generate many different realistic images, this is an undesirable outcome.  This can largely be prevented by promoting stability in the models.


General Approach

-Use convolutional neural networks because they frequently work well for image-based problems.

-Make a model complex enough to output realistic images, but simple enough to have reasonable training times.

-Slow down the Discriminator relative to the Generator so that the models stay relatively even in competition, which will maximize learning and prevent mode collapse.


Discriminator Summary (visible in more detail in the code)

4 convolution layers, with inputs normalized.  Leaky ReLU activation function after each convolution.  Flatten down to 512 nodes, and from there into a single activation layer, which will output a 1 or a 0.  Roughly 250,000 total parameters.


Generator Summary (visible in more detail in the code)
Begin with 2048 seeded parameters, reshape into 4x4x128 and normalize.  From there, perform 3 inverse convolutions, each normalized and activated with Leaky ReLU, eventually ending wiht a 32x32x3 output.  Roughly 750,000 total parameters (3 times as much as the Discriminator to give it an edge when learning).


Finer Model Details

-Image values scaled to be between -1 and 1 for normaliztion (rather than 0 and 255)

-Discriminator Learning Rate of 2e-4, Generator Learning Rate of 1e-4.  Literature results suggested using a smaller learning rate for the generator so that it can take smaller, more precise steps to fool the discriminator.  This also discourages it from making fast, imprecise, and unrealistic changes.  Overall it gives the generator a slight advantage in terms of precision, encouraging better competition between the two models.

-Optimizer selected was Adam.  Studies seem to show Adam has slightly better results than other options.

-Loss function selected was Binary Cross-Entropy.  Studies seemed to slow that the loss functions all performed equally well, so the selection doesn't matter.


Methods of Giving the Generator Advantages

1.  Label Smoothing
When giving 'answers' to the Discriminator, instead of giving scores of 1 and 0, I smoothed the values to be 0.8-1 or 0-0.1.  This helps reduce certainty in the Discriminator's guesses and helped prevent it from outcompeting the Generator.
2.  Ending Training Early
If the model was allowed to train for too long, the Discriminator ended up dominating and the Generator would do weird things to try to trick it, ultimately reducing the quailty of the generated images.  I started cutting off training at 150 epochs to prevent this, but fine-tuning this cutoff point for each of the 10 classes would definitely be a quick way to immediately improve the outputs of these models.


Training Time

On my machine, each epoch on the models that focused on an individual class took 3-4 minutes to train the ~1 million parameters.


Potential Improvements

-Larger models with more trainable parameters, given that additional computing power was obtainable

-Training on a GPU to decrease training time, allowing for expansion of the models

-Adding noise to the images, to further reduce Discriminator certainty

-Saving the models periodically or auto-stopping training when the Discriminator starts outperforming the Generator

-Additional iteration on the hyperparameters like the learning rates to better stabilize the models


