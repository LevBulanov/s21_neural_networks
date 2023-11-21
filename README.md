# Neural Networks

Summary: this project is an introduction to neural networks.

## Contents

1. [Chapter I. Preamble](#chapter-i-preamble) \
    1.1. [Introduction](#chapter-ii-introduction)
2. [Chapter II. Introduction](#chapter-ii-introduction) \
    2.1. [Feed Forward Neural Networks](#feed-forward-neural-networks) \
    2.2. [Universal Approximation Theorem](#universal-approximation-theorem) \
    2.3. [Backpropogation algorithm](#backpropagation-algorithm) \
    2.4. [Variants of SGD](#variants-of-sgd) \
    2.5. [Putting all together](#putting-it-all-together)
3. [Chapter III. Goal](#chapter-iii-goal)
4. [Chapter IV. Instructions](#chapter-iv-instructions)
5. [Chapter V. Task](#chapter-v-task)
6. [Chapter VI. Bonus part](#chapter-vi-bonus-part)

## Chapter I. Preamble

We have already discussed linear, tree-based and other models for supervised problems. Linear models make the strong assumption that the input-output mapping is linear; that class of models is interpretable yet has low accuracy in general.

One way to get a more accurate model is to switch to tree-based models that separate input space into regions leading to the ability of solving non-linear problems. Another way was shown in previous chapters: we extend the feature set with non-linear features such as products, fractions, polynoms, trigonometric or logarithmic variations of original features. This process requires manual creation and selection of features which is very limiting.

**Neural networks** and **deep learning** in general provide us with flexible models that automate the feature creation process and allow us to work with sophisticated non-linear problems.

## Chapter II. Introduction

### Feed Forward Neural Networks

Let’s refresh the definition of the Logistic Regression model.
Logistic Regression is model of the following form:

$$p(y=i|\mathbf{x}, \boldsymbol{\theta}) = \sigma (a) = \frac{1}{1 + e^{-a}}, \text{where } a = \mathbf{w}^T \mathbf{x} + b \text{ and }  \boldsymbol{\theta} = (\mathbf{w}, b)$$

Function $`\boldsymbol \sigma`$ is called **sigmoid**. Logistic Regression is equivalent to linear combination of features of matrix $`\boldsymbol X`$ and adding bias term **b**. Linear combination of features with sigmoid non-linearity can be viewed as an implicit feature generation process.
Putting another Logistic Regression classifier on top of the new feature space allows to model more complex functions:

$$p(y|\mathbf{x}, \boldsymbol{\theta}) = \text{Ber}(y|\sigma(\mathbf{a}_3))$$
$$\mathbf{a}_3 = \mathbf{w}_3^T \mathbf{z}_2 + b_3$$
$$\mathbf{z}_2 = \varphi(\mathbf{W}_2 \mathbf{z}_1 + b_2)$$
$$\mathbf{z}_1 = \varphi(\mathbf{W}_1 \mathbf{x} + b_1)$$


[Source](https://probml.github.io/pml-book/book1.html)

where Ber is [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).

Imagine for now that we have a way to learn all trainable (matrices **W** and biases **b**) parameters simultaneously. This model is known as a **feedforward neural network (FFNN)** or **multilayer perceptron (MLP)**. Second layer of the network trains the classifier and the first level learns feature transformation that helps the classifier to “separate” classes in new, “customized” feature space. We can stack more than 2 layers to receive more powerful features at the end. In addition, the sigmoid function can be replaced with another non-linear function. Non-linear functions in this context are called **activation functions**. Layers between input and output layers are called **hidden layers**. In the picture below, $`z_1`$ and $`z_2`$ are hidden layers.

Schematically, differences between neural and “non-neural” supervised learning is illustrated in the following picture: every consecutive layer transforms input into more and more sophisticated features that are better suited for solving original supervised problem. At the end we see a trainable classifier module (model similar to Logistic Regression).

![Deep learning](misc/images/deep_learning.png)

[Source](https://drive.google.com/file/d/1Q7LtZyIS1f3TfeTGll3aDtWygh3GAfCb/view)

Neural networks showed to be the most effective in domains with non-tabular (sometimes also called unstructured) data because of their compositionality nature. In computer vision problems, for example, features on different layers may be edges, corners, object parts, etc (look at the following pictures).

![Neural networks composition](misc/images/neural_networks_composition.png)

[Source](https://drive.google.com/file/d/1Q7LtZyIS1f3TfeTGll3aDtWygh3GAfCb/view)

We say (according to [Yann LeCun](https://drive.google.com/file/d/1Q7LtZyIS1f3TfeTGll3aDtWygh3GAfCb/view)) that the model is a **Deep Learning model** if it is a neural network and contains more than 1 hidden layer. Linear models, KNN, Naive Bayes don’t have hierarchy of features; CART aren’t considered deep because they operate in the original feature space; and only SVM’s can be viewed as a special case of MLP with 1 [hidden layer](https://www.deeplearningbook.org/contents/optimization.html).

MLP networks aren’t the only neural network architecture but they represent basic building blocks for more sophisticated neural network models.

### Universal Approximation Theorem

Geometrically, linear transformation (matrix multiplication and adding the bias term) can move, reflect and scale inputs in space, non-linear function “squashes” inputs. These two types of transformations give an opportunity (in theory) to model any arbitrary complex function. It is a mathematically proven result called **Cybenko Theorem** (or **Universal Approximation Theorem**). [Source](https://en.wikipedia.org/wiki/Universal_approximation_theorem):

> MLP with 1 or more hidden layers and sigmoid activation functions are capable of accurately representing any continuous function (it can model any continuous function, given enough hidden units, to any desired level of accuracy).

Additional results were proved after Cybenko Theorem: any differentiable non-linear non-polynomial activation can be used in MLP to produce an Universal approximator. One of the most popular choices is **rectified linear unit**:

```math
\text{ReLU}(\boldsymbol{x}) = \max(0, \boldsymbol{x})
```

Unfortunately, Cybenko Theorem doesn’t show the way to learn optimal parameters (weights) for MLP layers. In addition, for good accuracy sometimes we need a very “wide” hidden layer: number of needed parameters can be exponential of size of the input.
These facts show that using MLP with more than 1 hidden layer is justified. We can think of it as decomposition of an extremely wide hidden layer into a sequence of smaller layers that build “hierarchy” of features.

### Backpropagation algorithm

Neural networks are very flexible because of their compositionality. We have to come up with a training algorithm that will allow training all layers at the same time.
Surprisingly, algorithms for training neural networks resemble learning procedures for linear models.

Applying MLP to some input means a chain of calculations: we use the first hidden layer and non-linearity to compute input for the second hidden layer, then use the output of the second layer after non-linearity as input for the third layer, etc. This process is called **forward propagation**.

If we use differentiable activation functions at every layer we can compute the gradient of loss function with respect to any parameter of the network. Computation of the gradient goes backwards:

1. Calculate gradient of the loss with respect to output layer’s parameters.
2. Calculate gradient of output layer’s parameters with respect to (w.r.t) previous layers’s parameters.
3. Using results of 1 and 2, apply chain rule theorem* and compute gradient of the loss w.r.t to parameters of the previous (for the output) layer.
4. Traversing from end of the network to the start we compute consequently partial derivatives of loss function w.r.t to every parameter of the model.

This algorithm is called **backpropagation**. Gradient computation in general form requires **matrix differentiation*** but the concept stays the same.

![Backpropagation](misc/images/backpropagation.png)

[Source](https://www.baeldung.com/cs/deep-learning-bias-backpropagation)

Having gradient of loss w.r.t to every learnable parameters allows to apply standard SGD (Stochastic Gradient Descent) procedure to train the network. In essence, training a neural network is similar to training linear models: computation of gradient is just more complicated. Backpropagation is clearly described [here](https://en.wikipedia.org/wiki/Backpropagation) and [here](https://www.baeldung.com/cs/deep-learning-bias-backpropagation).

Training neural networks is simple in theory, but complicated in practice. There are various difficulties related to training:

- Parameter initialization. There are different strategies for random initialization.
- Gradient explosion. Gradients can be too large and mislead our optimization process badly.
- Gradient saturation. On the other hand, gradients can be too small (because improper activation functions, poor initialization, etc.) and our optimization process extremely slows down and eventually stops.

[Source](https://d2l.ai/) contains an overview of modern tricks and tips for training deep networks.

### Variants of SGD

Loss functions in neural networks depend on thousands and millions of parameters, making loss surface highly non-convex. Sometimes, SGD algorithm is not enough to receive a good quality because of the existence of local minimas or saddle points in the loss surface. There are various tricks that improve vanilla SGD, but the most basic are:

- Momentum: accumulate an exponentially decaying moving average of part gradients.
- Adaptive learning rate: different learning rates for different parameters, we go fast in some directions and slow in others, based on amplitude of the gradient.

SGD and its variations in the context of optimizing neural networks are called **optimizers**.
[Adam algorithm](https://www.deeplearningbook.org/contents/optimization.html) is considered to be the best out-of-the-box choice.
Following pictures show the performance of the **SGD** algorithm, [**Adam** and **QHAdam**](https://arxiv.org/abs/1810.06801) on optimizing 2 sophisticated functions(details are [here](https://github.com/jettify/pytorch-optimizer)).

SGD  
![SGD](misc/images/SGD.png)

Adam  
![Adam](misc/images/Adam.png)

We see that SGD failed in both cases; Adam has success but with difficulties and a very noisy path.

QHAdam  
![QHAdam](misc/images/QHAdam.png)

QHAdam is a modification of the original Adam method. It can be seen that it failed in the first case but performed very well in the second case, having a nice smooth path to the global minima.

From these examples we understand that in different situations and problems optimizers may have different performance.

### Putting it all together

Neural networks reduce feature engineering routines performed by humans but replace part of that time with network engineering: process of designing and training neural networks. We have to select the proper number of layers, activation functions, optimizer, etc.

[This resource](http://playground.tensorflow.org/) gives a playground simulator on network engineering for solving non-linear classification problems.

[This](http://d2l.ai/) contains a detailed overview of neural networks and deep learning in general.

## Chapter III. Goal

The goal of this task is to get a deep understanding of the basics of neural networks.

## Chapter IV. Instructions

- This project will only be evaluated by humans. You are free to organize and name
your files as you desire.
- Here and further we use Python 3 as the only correct version of Python.
- For training deep learning algorithms you can try Google Colab. It offers kernels
(Runtime) with GPU for free that is faster than CPU for such tasks.
- The norm is not applied to this project. Nevertheless, you are asked to be clear
and structured in the conception of your source code.
- Store the datasets in the subfolder data.

## Chapter V. Task

1. Download data from [Don’tGetKicked competition](https://www.kaggle.com/c/DontGetKicked).

    Design train/validation/test split.

    Use “PurchDate” field for splitting, test must be later in time than validation, the same goes for validation and train: train.PurchDate < valid.PurchDate < test.PurchDate.
    Use the first 33% of dates for train, last 33% of dates for test, and middle 33% for validation set. *Don’t use the test dataset until the end!*

    Use LabelEncoder or OneHotEncoder from sklearn to preprocess categorical variables. Be careful with data leakage (fit Encoder on train and apply on validation & test). Consider another encoding approach if you meet new categorical values in valid & test (unseen in the training dataset). [Example](https://contrib.scikit-learn.org/category_encoders/count.html)
2. Create a Python class, MLP with 1 hidden layer and sigmoid activation function at the end of the network.
It should support *fit, predict_proba* and *predict methods*. Also, the number of neurons in the hidden layer and activation function must be parameters of your class.

    Here is the blueprint:

    ```python
    model = MLP(n_hidden=100, activation=np.arctan)
    model.fit(Xtrain, ytrain)
    model.predict_proba(Xvalid)
    ```

    - Initialize the network with small random numbers.
    - Use Log Loss (Binary Cross Entropy) as the loss function.
    - Implement forward pass; you can use fixed batch-size like 32, forward pass maps arrays of shape (batch_size, number_of_features) to arrays of shape (batch_size, 2) where 2 means dimensions of [probability_of_0, probability_of_1] output space.
    - Write down gradients of loss function w.r.t to parameters of the network. Use [4](https://en.wikipedia.org/wiki/Backpropagation), [9](http://d2l.ai/) as a guide for gradient derivation.
    - Use gradients to perform Backprop. Implement backward pass.
    - Implement SGD algorithm to tune parameters of the model.
    - Design a basic train-validation loop: iterate over train dataset, batch by batch, update parameters of the network and check quality of the model using validation set.
    - Create code for updating weights of the network using SGD or Adam.
3. With your MLP module and careful network engineering you have to receive at least 0.15 Gini score on the validation dataset. You can train for more than 1 epoch, use various activation functions, and different optimizers (like SGD or Adam).
4. Use MLPClassifier from sklearn and check its performance on validation dataset. Is it better than your module? If it is, why?
5. Implement and try different activation functions: sigmoid, ReLU, cosine. Remember you have to derive gradients for every different activation function. What activation function gives the best performance on the validation dataset?
6. Design MLP module with 1 hidder layer using any high-level deep learning framework: Pytorch, Keras or Tensorflow. Check its performance on validation dataset.  Is it better than your module? If it is, why?
7. Pick up the best model and estimate its performance on the test dataset:  check Gini scores on all three datasets for your best model: train Gini, valid Gini, test Gini. Can you see any drop in performance when comparing valid quality vs test quality? Is your model overfitted or not? Explain.

## Chapter VI. Bonus part

1. Implement Adam algorithm to tune parameters of the model.

Chain rule is explained in [source](https://en.wikipedia.org/wiki/Chain_rule)

Matrix differentiation is cracked in the following [source](http://d2l.ai/chapter_preliminaries/autograd.html)

>Please leave feedback on the project in the [feedback form.](https://forms.yandex.ru/cloud/646b479ac769f12ef19bce19/) 
