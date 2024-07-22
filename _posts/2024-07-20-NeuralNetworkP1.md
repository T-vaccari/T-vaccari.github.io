---
layout: post
title: "Neural Network : Part One"
categories: jekyll update
---

# Introduction

In this post, I would like to introduce how neural networks are built under the hood. I will explore the essential components and mechanisms that enable neural networks to learn and make predictions. By the end of this article, you should have a clearer understanding of how these powerful models work under the hood.


# Overview of Neural Networks

Before starting to deep dive into neural network I would like to roughly remember to the reader what is the main goal and how it's reached. Neural networks (NNs) are a class of machine learning models inspired by the structure and function of the human brain. The primary objective of a neural network is to learn a mapping from input data to desired output values by adjusting its parameters through a process of optimization.

## 1. Goal of neural networks
The central aim of a neural network is to learn how to produce accurate outputs when given specific inputs. This process involves training the network on a dataset, where each data point consists of an input and a corresponding target output. For instance, in a classification problem, the input might be an image, and the target output would be the label of the object in the image. Or maybe in a regression problem we want to train our model to answer the question how much?.

## 2. Training Process

### 2.1 Feeding data into the net and comparison of the output
During training, the network is provided with a set of input data and its corresponding target outputs. The network makes predictions based on the input data, which are then compared to the target outputs. This comparison is crucial for evaluating how well the network is performing.
### 2.2 Loss Function
To quantify the difference between the network's predictions and the target outputs, we use a loss function (or cost function). The loss function measures the prediction error. Common loss functions include Mean Squared Error for regression tasks and Cross-Entropy Loss for classification tasks. A lower value of the loss function indicates better performance of the network.
### 2.3 Optimization via backpropagation
The goal of training is to minimize the loss function, which involves finding the optimal set of parameters (weights and biases) for the network. To achieve this, we use an optimization algorithm that adjusts the parameters to reduce the loss.
Backpropagation is the method used to compute the gradients of the loss function with respect to the network's parameters. 
It involves:
Forward Pass: Computing the predictions of the network and the loss.
Backward Pass: Calculating the gradients of the loss function with respect to each parameter using the chain rule of calculus.
### 2.4 Gradient Descent
Once the gradients are computed, they are used by the optimization algorithm (often gradient descent or its variants) to update the network's parameters. The network parameters are adjusted in the direction that reduces the loss function. This process is iterated over multiple epochs (passes through the entire dataset) until the loss function converges to a minimum value or sufficiently small error.

Roughly speaking this is the process that we want to follow for achieving our goal.
With the foundational concepts established, we will now delve into the detailed steps required to build a neural network. Firstly, we need to create a class that represents data within our neural network. This class must track the origin of each data value, including the operations and values used to compute it, in order to facilitate accurate gradient calculation during backpropagation. We are goin to call the class for representing data *Value* class.


# Breakdown of the code for the Value class
Now I will start to implement in code what we have seen in the foundational concepts about NN.

## Building the basic item of the class and the attribute of the object
``` python
class Value: 
    
    def __init__(self, data, _children = (), _op = '' ):
        
        self.data = data
        self._previuosly = set(_children)
        self._operations = _op
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"

```

  Here we initialize the value objects with it's own attribute. Let's breakdown each attribute :
  - self.data = data, saves the value of the object
  - self._previuosly = set(_children), saves in a set the children of it's value, that means that we save what values generated this value. Thanks to the set we can avoid duplicates
  - self._operations = _op, we keep track of what operation generated this value
  - self.grad = 0.0 , we initially set it to zero, then we are going to modify it accordingly to the rules of backpropagation, we store it in this attribute
  - self._backward = lambda : None , we need to store here the function that provide to us the gradient, we set it initially to a lambda None because it dipends on the operation that generated this Value object.


The __repr__ method is used to print the value object.
So for example if we want to generate a value object we can do as follow :
   
``` python

    a = Value(3)
    print(a)
    >>> Value(data = 3)

```

## Building the basic operation and the relative backward function for value object

```python
class Value : 
    ... #Code seen before

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
    
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
    
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
      
        return out


```

Here we have the code for doing addition and multiplication.
Simply in the method for addition we create an out Value object featured by a new value and a set of children(the two Value object that originated it) and then we append the operation that originated that new Value object. Then from the foundamental of calculus we know that the gradient of the two children due to this operation can be calculated as shown in the code. For much more detail watch gradient explained [here](https://en.wikipedia.org/wiki/Gradient). Let's watch how does the add function behave : 

``` python
    a = Value(3)
    b = Value(4)
    c = a + b
    print(c)
    >>> Value(data = 7)
    print(c._previously)
    >>> (Value(data = 3),Value(data = 4))
    print(c._operation)
    >>> '+'

```

__! Currently finishing the post !__



