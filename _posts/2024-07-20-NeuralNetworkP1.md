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

    
```

Here we have the code for the __add__ method.
Simply in this method  we create an out Value object featured by a new value and a set of children(the two Value object that originated it) and then we append the operation that originated that new Value object. Then from the foundamental of calculus we know that the gradient of the two children due to this operation can be calculated as shown in the code. For much more detail watch gradient explained [here](https://en.wikipedia.org/wiki/Gradient). Let's watch how does the add function behave : 

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

Now let's look a the code for the __mul__ method :

``` python
class Value :
    ... #Code seen before

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

```

In the __add__ method, we create a new Value object with a new value and a set of children, which are the two Value objects that originated this result. We also append the operation that produced this new Value object. The backward function is designed to compute the gradient according to calculus rules.

Note that we use the += operator when updating the .grad attribute. This is because we want to accumulate the gradient. For example, if the Value object is involved in multiple operations, the gradient should reflect all these operations, making the gradient cumulative.
Generally this is the process that we follow to create new method for operation for thi class. For the purpose of creating a neural network we need also a function that can work for us as an [activation function](https://en.wikipedia.org/wiki/Activation_function).
In this case we add the tanh function for this scope

``` python
class Value:
   ... #Code seen before
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

        out = Value(t, (self, ),'tanh')
    
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        
        return out

```
From calculus we know that tanh is defined as : 

$$
\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}
$$

Then we can implement the backward function knowing that : 

$$
\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)
$$

Now we implement the expnonential method for being able of using tanh: 

```python
class Value:
    ...#Code seen before
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        
        def _backward():
            self.grad += out.data * out.grad 
        out._backward = _backward
        
        return out

```


We have now implemented most of the basic operations for our Value object. The final step is to implement a backward method. This method will be responsible for calling the backward functions of each object in the proper order.

## Building the backward method

Here's the code for the function : 
```python
class Value : 

    ... #Code seen before

    def backward(self):
    
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._previously:
                build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


```
Thanks to the implementation of the Value object, we are able to construct a computation graph that captures the entire history of how a Value object was derived. This graph is formed by following the _previously set of each Value object, which tracks all the predecessor values involved in its computation.

To compute the gradients, we begin by focusing on the final Value object, which typically results from a loss function in a neural network. Since we are interested in the gradient of this final Value with respect to itself, we initialize its gradient (self.grad) to 1. This initialization signifies that the gradient of the final Value with respect to itself is 1.

Following this, we execute the backward pass by invoking the _backward function for each Value object. This process starts with the final Value and proceeds backward through the computation graph to the initial Value objects. This reverse traversal ensures that the gradient for each Value is computed correctly based on the chain rule of differentiation, allowing us to accumulate the gradients appropriately.

We've now looked at the basics of how a neural network works. This simple approach helps us understand the core concepts behind its operation.
Now it's time to create a very simple net.

## Creating a Neural Network
Now we will look more deeply on how to create a net by little steps.
### Create one artifical neuron
Now we can create an [artifcial neuron](https://en.wikipedia.org/wiki/Artificial_neuron) with two inputs.
This is an example of how does a neuron work : 
```python
    #Inputs x1,x1
    x1 = Value(1.0)
    x2 = Value(2.0)
    #Weights w1,w2
    w1 = Value(6.0)
    w2 = Value(-4.0)
    # bias of the neuron
    b = Value(9.8)
    # x1*w1 + x2*w2 + b
    x1w1 = x1*w1
    x2w2 = x2*w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    # Activation function
    o = n.tanh()

```

o is the output of our neuron. The next step is to define a the class Neuron, here's the code:
```python

class Neuron:
  
  def __init__(self, nin):
    self.w = list()
    self.b = Value(random.uniform(-1,1))

    for _ in range (nin):
        self.w.append(Value(random.uniform(-1,1)))
  
  def __call__(self, x):
    # w * x + b
    activation = 0
    for wi,xi in zip(self.w,x):
        activation += wi*xi
    activation+=self.b
    out = activation.tanh()

    return out
  
  def parameters(self): #For storing the parameters
    return self.w + [self.b]


```
Here we define the class Neuron, in the __init__ method we create the weights based on the number of input(nin) and then we create the bias Value.
When we call the Neuron it performs the activation based on the input and gives in output the result. Here's an example of usage
```python
x = [1,2,3]
a = Neuron(3)
print(a(x))
>>> Value(data = something)

```

### Create a layer
The next step is to create a layer made of neuron, here's the code:

```python 

class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = list()

    for _ in range(nout):
        self.neurons.append(Neuron(nin))
  
  def __call__(self, x):
    outs = list()
    
    for neuron in self.neurons:
        outs.append(neuron(x))

    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

```

In the __init__ method we initialize a layer specifing the number of input for each neuron(nin) and the number of ouput for the layer(nout).
When we call it, the __call__ method push the input to every neuron and return the out of every neuron.

### Create the multilayer perceptron
The final step is to create an[ MLP fully connected](https://en.wikipedia.org/wiki/Multilayer_perceptron), here's the code : 

```python

class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = list()

    for i in range(len(nouts)):
        self.layers.append(Layer(sz[i],sz[i+1]))
    
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

```

In the __init__ method as usual we initialize the MLP giving the number of inputs of the neurons(nin) and then a list(nouts) where we store the number of neurons for each layer. When we creare the self.layer attribute we iterate over a list that we create where we have the number of inputs and outputs require for every layer.
Now let's give a look of how we can use it :

```python
x = [1.0, 5.0, -6.0]
net = MLP(3, [4, 4, 1])
net(x)
>>>Value(data=something)


```
Now we are ready to train the net that we have built.
### Creating the dataset
To start we can manually create a very easy dataset and the desired target :

```python
xs = [
  [1.0, 2.0, -1.0],
  [6.0, -4.0, 1],
  [2, 1.0, -1.0],
  [4.0, 3.0, -2.0],
]
ys = [2.0, -4.0, -3.0, 2.0]


```
So when we feed to our net the first list we want to obtain as a result the firs element of the ys list.
Now all we need is a loss function that can tell us how good are the output of our net.
When we have our loss function we can call the backpropagation on it(remember that is a Value object so we can do it) and then we can slighly adjust the parameters according to what minimize the loss function.
Now we can implement it :
```python
for k in range(20):
  
    # forward pass
    ypred = list()
    for x in xs:
    ypred.append(net(x))
    loss = 0

    for ygt,yout in zip(ys,ypred):
        loss+= (yout-ygt)**2

    # backward pass
    for p in net.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
  
```
In this piece of code we train our net following this step :

1. Forward pass
   We feed to the net the dataset and then we save the output of the net. Then we evaluate the loss using [MSE function](https://en.wikipedia.org/wiki/Mean_squared_error). 
2. Backward Pass
   We reset all the gradient of all parameters and then we compute the new grad of each parameter calling backward on the loss function.
3. Update
   We update the parameters in the direction that minimize the loss function using a learning rate of 0.1

# Conclusion
In this first part we built the foundamental for the understading of neural network. In the next part we are goin to see how to affine our techniques.








