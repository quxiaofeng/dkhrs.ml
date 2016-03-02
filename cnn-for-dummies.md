---
layout: default
permalink: /cnn-for-dummies/
---

# CNN for Dummies #

Written by Jianxin WU

Translated by Xiaofeng QU

## Abstract ##

This is a note that describes how a Convolutional Neural Network (CNN) operates from a mathematical perspective. This note is self-contained, and the focus is to make it comprehensible to beginners in the CNN field.

## Contents ##

1. [Motivation](#motivation)
2. [Preliminaries](#preliminaries)
   1. [Tensor and vectorization](#tensor-and-vectorization)
   2. [Vector calculus and chain rule](#vector-calculus-and-chain-rule)
3. [CNN in a nutshell](#cnn-in-a-nutshell)
   1. [The architecture](#the-architecture)
   2. [The forward run](#the-forward-run)
   3. [Stochastic gradient descent (SGD)](#stochastic-gradient-descent-sgd)
   4. [Error back propagation](#error-back-propagation)
4. [The convolution layer](#the-convolution-layer)
   1. [Input, output, filters, and notations](#input-output-filters-and-notations)
   2. The (forward) convolution
   3. Expanding the convolution
   4. Now let's make it formal
   5. The Kronecker product
   6. Backward propagation: the parameters
   7. Even higher dimensional indicator matrices
   8. Backward propagation: the supervision signal 
5. The pooling layer
6. The reverse operators
7. The ReLU layer
8. Conclusions

## 1 Motivation ##

The Convolutional Neural Network (CNN) has shown excellent performance in many computer vision and machine learning problems. Many solid papers have been published in this topic, and quite some high quality open source CNN software packages have been made available.

There are also well-written CNN tutorials or CNN software manuals. However, I believe that an introductory CNN material specifically prepared for beginners is still needed. Research papers are usually very terse and lack details. It might be difficult for beginners to read such papers. A tutorial may not cover all the necessary details to understand exactly how a CNN runs (i.e., learning and predicting).

This note tries to present beginners with a document that

+ Is self-contained. It is expected that all required mathematical background knowledge are introduced in this note itself;
+ Has details for all the derivations. As the title ("for dummies") suggests, this note tries to explain all the necessary math in details. We try not to ignore any step in a derivation. Thus, it should be easy for a beginner to follow (although an expert may feel this note as tautological.)
+ Ignores implementation details. The purpose is for a reader to understand how a CNN runs at the mathematical level. We will ignore those implementation details. In CNN, making correct choices for various details is one of the keys to its high accuracy. However, we intentionally left this part out, in order for the reader to focus on the mathematics. After understanding the mathematical principles and details, it is more advantageous to learn these implementation and design details with hands-on experience by running real CNN codes.

This note is modeled after [Vedaldi and Lenc, 2014](http://arxiv.org/abs/1412.4564).

## 2 Preliminaries ##

We start by a discussion of some background knowledge that are necessary in
order to understand how a CNN runs. One can ignore this section if he/she is
familiar with these basics.

### 2.1 Tensor and vectorization ###

Everybody is familiar with vectors and matrices.
We use a symbol shown in boldface to represent a vector, e.g., \\( \\mathbf{x} \\in \\mathbb{R}^D \\) is a column vector with \\( D \\) components.
We use a capital letter to denote a matrix, e.g., \\( X \\in \\mathbb{R}^{H\\times{}W} \\) is a matrix with \\( H \\) rows and \\( H \\) columns. The vector \\( \\mathbf{x} \\) can also be viewed as a matrix with \\( 1 \\) column and \\( D \\) rows.

These concepts can be generalized to higher-order matrices, i.e., tensors.
For example, \\( \\mathbf{x} \\in \\mathbb{R}^{H\\times{}W\\times{}D} \\) is a 3D tensor.
It has \\( HWD \\) numbers, and each of them can be indexed by an index triplet \\( (i, j, d) \\), with \\( 0 \\leq i \< H \\), \\( 0 \\leq j \< W \\), and \\( 0 \\leq d \< D \\).
Another way to view a 3D tensor is to treat it as containing \\( D \\) slices of matrices.
Every slice is a matrix with size \\( H \\times W \\).
The first slice contains all the numbers in the tensor that are indexed by \\( (i,j,0) \\).
When \\( D = 1 \\), a 3D tensor reduces to a matrix.

Higher order matrices also exist.
For example, we will soon see that the filter bank in a convolution layer in a CNN is a 4D tensor.

Given a tensor, we can arrange all the numbers inside it into a long vector, following a pre-specified order.
For example, in Matlab, the (\\(:\\)) operator converts a matrix into a column vector in the column-first order.
An example is:

<a name="eqn_tensor"></a>
\\[
A = \\left(\\begin{matrix}
1 & 2 \\\
3 & 4
\\end{matrix}\\right), \\qquad
A(:) = (1,3,2,4)^T = \\left(\\begin{matrix}
1 \\\
2 \\\
3 \\\
4
\\end{matrix}\\right) \\qquad (1)
\\]

In mathematics, we use the notation "\\( vec \\)" to represent this vectorization operator.
That is, \\( vec(A) = (1, 3, 2, 4)^T \\) in the example in [Equation 1](#eqn_tensor).
In order to vectorize a 3D tensor, we could vectorize its first slice (which is a matrix), then the second slice, . . . , till all slices are vectorized.
The vectorization of the 3D tensor is then the concatenation of all the slices.
This recursive process can be applied to vectorize a 4D (or even higher order) tensor.

### 2.2 Vector calculus and chain rule ###

The CNN learning process depends on vector calculus and chain rule.
Suppose \\( z \\) is a scalar (i.e., \\( z \\in \\mathbb{R} \\)) and \\( \\mathbf{y} \\in \\mathbb{R}^H \\) is a vector.
If \\( z \\) is a function of \\( \\mathbf{y} \\), then the partial derivative of \\( z \\) with respect to \\( \\mathbf{y} \\) is a vector, defined as
\\[
\\qquad \\qquad \\qquad \\left( \\frac{\\partial z}{\\partial \\mathbf{y}} \\right)\_i = \\frac{\\partial z}{\\partial y\_i}. \\qquad \\qquad \\qquad (2)
\\]
In other words, \\( \\frac{\\partial z}{\\partial \\mathbf{y}} \\) is a vector having *the same size* as \\( \\mathbf{y} \\), and its \\( i \\)-th element is \\( \\frac{\\partial z}{\\partial y\_i} \\).
Also note that \\( \\frac{\\partial z}{\\partial \\mathbf{y}^T} = \\left( \\frac{\\partial z}{\\partial \\mathbf{y}} \\right)^T \\).

Furthermore, suppose \\( \\mathbf{x} \\in \\mathbb{R}^W \\) is another vector, and \\( \\mathbf{y} \\) is a function of \\( \\mathbf{x} \\).
Then, the partial derivative of \\( \\mathbf{y} \\) with respect to \\( \\mathbf{x} \\) is defined as
\\[
\\qquad \\qquad \\qquad \\left( \\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}^T} \\right) = \\frac{\\partial y\_i}{\\partial x\_j} \\qquad \\qquad \\qquad (3)
\\]
This partial derivative is a \\( H \\times W \\) matrix, whose entry at the intersection of the \\( i \\)-th row and \\( j \\)-th column is \\( \\frac{\\partial y\_i}{\\partial x\_j} \\).

It is easy to see that \\( z \\) is a function of \\( \\mathbf{x} \\) in a chain-like argument: a function maps \\( \\mathbf{x} \\) to \\( \\mathbf{y} \\), and another function maps \\( \\mathbf{y} \\) to \\( z \\).
A chain rule can be used to compute \\( \\frac{\\partial z}{\\partial \\mathbf{x}^T} \\), as
<a name="eqn_chain_rule"></a>\\[
\\qquad \\qquad \\qquad \\frac{\\partial z}{\\partial \\mathbf{x}^T} = \\frac{\\partial z}{\\partial \\mathbf{y}^T} \\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}^T}. \\qquad \\qquad \\qquad (4)
\\]

A sanity check for [Equation 4](#eqn_chain_rule) is to check the matrix/vector dimensions.
Note that \\( \\frac{\\partial z}{\\partial \\mathbf{y}^T} \\) is a row vector with \\( H \\) elements, or a \\( 1 \\times H \\) matrix (Be reminded that \\( \\frac{\\partial z}{\\partial \\mathbf{y}} \\) is a column vector).
Since \\( \\frac{\\partial \\mathbf{y}}{\\partial \\mathbf{x}^T} \\) is an \\( H \\times W \\) matrix, the vector/matrix multiplication between them is valid, and the result should be a row vector with \\( W \\) elements, which matches the dimensionality of \\( \\frac{\\partial z}{\\partial \\mathbf{x}^T} \\).

For specific rules to calculate partial derivatives of vectors and matrices, please refer to [the Matrix Cookbook](http://www2.imm.dtu.dk/pubdb/views/edoc_download.php/3274/pdf/imm3274.pdf).

## 3 CNN in a nutshell ##

In this section, we will see how a CNN trains and predicts in an abstract level, with the details leaved out for later sections.

### 3.1 The architecture ###

A CNN usually takes a 3D tensor as its input, e.g., an image with \\( H \\) rows, \\( W \\) columns, and 3 slices or channels (R, G, B color channels).
The input then sequentially goes through a series of processing.
One processing step is usually called a layer, which could be a convolution layer, a pooling layer, a normalization layer, a fully connected layer, a loss layer, etc.
We will introduce the details of these layers later in this note.

For now, let us give an abstract description of the CNN structure first.


<a name="eqn_abstract_cnn_structure"></a>
\\[
\\mathbf{x}^1 \\rightarrow \\left\[\\mathbf{w}^1\\right\] \\rightarrow \\mathbf{x}^2 \\rightarrow \\ldots \\rightarrow \\mathbf{x}^{L-1} \\rightarrow \\left\[\\mathbf{w}^{L-1}\\right\] \\rightarrow \\mathbf{x}^L \\rightarrow \\left\[\\mathbf{w}^L\\right\] \\rightarrow \\mathbf{z} \\qquad (5) 
\\]

The above [Equation 5](#eqn_abstract_cnn_structure) illustrates how a CNN runs layer by layer forward.
The input is \\( \\mathbf{x}^1 \\), usually an image (3D tensor). It goes through the processing in the first layer, which is the first box.
We denote the parameters involved in the first layer's processing collectively as a tensor \\( \\mathbf{w}^1 \\).
The output of the first layer is \\( \\mathbf{x}^2 \\), which also acts as the input to the second layer processing.

This processing proceeds till all layers in the CNN has been finished, which outputs \\( \\mathbf{x}^L \\).
One additional layer, however, is added for backward error propagation.
Let’s suppose the problem at hand is an image classification problem with \\( C \\) classes.
A commonly used strategy is to output \\( \\mathbf{x}^L \\) as a \\( C \\) dimensional vector, whose \\( i \\)-th entry encodes the prediction \\( P \\left(c\_i \\mid \\mathbf{x}^1 \\right) \\).
That is, given the input \\( \\mathbf{x}^1 \\), \\( \\mathbf{x}^L \\) outputs a vector of CNN-estimated posterior probabilities.

Let's suppose \\( \\mathbf{t} \\) is the corresponding target (ground-truth) value for the input
\\( \\mathbf{x}^1 \\), then a cost or loss function can be used to measure the discrepancy between the CNN prediction \\( \\mathbf{x}^L \\) and the target \\( \\mathbf{t} \\).
For example, the simplest loss function could be
<a name="eqn_simplest_loss"></a>\\[
\\qquad \\qquad \\qquad z = \\| \\mathbf{t} - \\mathbf{x}^L \\|^2, \\qquad \\qquad (6)
\\]
although more complex loss functions are usually used.

For simplicity, [Equation 5](#eqn_abstract_cnn_structure) explicitly models the loss function as a loss layer, whose processing is modeled as a box with parameters \\( \\mathbf{w}^L \\).

Note that some layers may not have any parameters, that is, \\( \\mathbf{w}^i \\) may be empty for some \\( i \\).

### 3.2 The forward run ###

Suppose all the parameters of a CNN model \\( \\mathbf{w}^1, \\ldots, \\mathbf{w}^{L-1} \\) has been learned, then we are ready to use this model for prediction.

Prediction only involves running the CNN model forward, i.e., in the direction of the arrows in [Equation 5](#eqn_abstract_cnn_structure).

Let's take the image classification problem as an example.
Starting from the input \\( \\mathbf{x}^1 \\), we make it pass the processing of the first layer (the box with parameters \\( \\mathbf{w}^1 \\)), and get \\( \\mathbf{x}^2 \\). In turn, \\( \\mathbf{x}^2 \\) is passed into the second layer, etc.
Finally, we achieve \\( \\mathbf{x}^L \\in \\mathbb{R}^C \\), which estimates the posterior probabilities of \\( \\mathbf{x}^1 \\) belonging to the \\( C \\) categories.
We can output the CNN prediction as
\\[
\\qquad \\qquad \\qquad \\arg\\max\_i \, \\mathbf{x}^L\_i. \\qquad \\qquad (7)
\\]

The problem is: how do we learn the model parameters?

### 3.3 Stochastic gradient descent (SGD) ###

Stochastic gradient descent (SGD) is the mainstream method to learn a CNN's parameters.
Let's suppose one training example \\( \\mathbf{x}^1 \\) is given for training such parameters.
The training process involves running the CNN network in both directions.
We first run the forward network to get \\( \\mathbf{x}^L \\) to achieve a prediction using the current CNN model.
Instead of outputting a prediction, we need to compare the prediction with the target \\( \\mathbf{t} \\) corresponding to \\( \\mathbf{x}^1 \\), that is, continue running the forward pass till the last loss layer.
Finally, we achieve a loss \\( z \\).

The loss \\( z \\) is then a supervision signal, guiding how the parameters of the model should be modified.
And the SGD way of modifying the parameters are
<a name="eqn_sgd"></a>\\[
\\qquad \\qquad \\qquad \\mathbf{w}^i \\leftarrow \\mathbf{w}^i - \\eta \\frac{\\partial z}{\\partial \\mathbf{w}^i}. \\qquad \\qquad (8)
\\]

In [Equation 8](#eqn_sgd), \\( \\eta \\) is a proper learning rate.

A *cautious note* about the notation.
In most CNN materials, a superscript indicates the "time" (e.g., training epochs).
But in this note, we use the superscript to denote the layer index.
Please do not get confused.
We do not use an additional index variable \\( t \\) to represent time.
In [Equation 8](#eqn_sgd), the \\( \\leftarrow \\) sign implicitly indicates that the parameters \\( \\mathbf{w}^i \\) (of the \\( i \\)-layer) are updated from time \\( t \\) to \\( t + 1 \\).
If an time index \\( t \\) is explicit used, this equation will look like
\\[
\\qquad \\qquad \\left( \\mathbf{w}^i \\right)^{t + 1} = \\left( \\mathbf{w}^i \\right)^t - \\eta \\frac{\\partial z}{\\partial \\left( \\mathbf{w}^i \\right)^t}. \\qquad (9)
\\]

A new problem now becomes apparent: how to compute the (partial) derivatives, which seem very complex?

### 3.4 Error back propagation ###

The last layer partial derivatives are easy to compute.
Because \\( \\mathbf{x}^L \\) is connected to \\( z \\) directly under the control of parameters \\( \\mathbf{w}^L \\), it is easy to compute \\( \\frac{\\partial z}{\\partial \\mathbf{w}^L} \\).
This step is only neede when \\( \\mathbf{w}^L \\) is not empty.
In the same spirit, it is also easy to compute \\( \\frac{\\partial z}{\\partial \\mathbf{x}^L} \\).

In fact, for every layer, we compute two sets of results: the partial derivatives of \\( z \\) with respect to the layer parameters \\( \\mathbf{w}^i \\), and that layer's input \\( \\mathbf{x}^i \\).

+ The term \\( \\frac{\\partial z}{\\partial \\mathbf{w}^i} \\), as seen in [Equation 8](#eqn_sgd), can be used to update the current (\\( i \\)-th) layer's parameters;
+ The term \\( \\frac{\\partial z}{\\partial \\mathbf{x}^i} \\) can be used to update parameters backwards, e.g., to the (\\( i \\)-1)-th layer. An intuitive explanation is that \\( \\frac{\\partial z}{\\partial \\mathbf{x}^i} \\) is the part of the "error" supervision information propagated from \\( z \\) backwards till the current layer, in a layer by layer fashion. Thus, we can continue the back propagation process, and use \\( \\frac{\\partial z}{\\partial \\mathbf{x}^i} \\) to both guide the updating of parameters and propagate the errors backwards to the (\\( i \\)-1)-th layer.

Thus, at each layer \\( i \\), we need to compute two sets of derivatives: \\( \\frac{\\partial z}{\\partial \\mathbf{w}^i} \\) and \\( \\frac{\\partial z}{\\partial \\mathbf{x}^i} \\).
This layer-by-layer backward updating procedure makes learning a CNN much easier.

Let's take the \\( i \\)-th layer as an example.
When we are updating the \\( i \\)-th layer, the back propagation process for the (\\( i \\)+1)-th layer must have been finished.
That is, we already computed the terms \\( \\frac{\\partial z}{\\partial \\mathbf{w}^{i+1}} \\) and \\( \\frac{\\partial z}{\\partial \\mathbf{x}^{i+1}} \\).
Both are stored in memory and ready for use.

Now our task is to compute \\( \\frac{\\partial z}{\\partial \\mathbf{w}^i} \\) and \\( \\frac{\\partial z}{\\partial \\mathbf{x}^i} \\).
Using the chain rule, we have
\\[
\\frac{}{} = \\frac{}{} \\frac{}{}, (10)
\\]
\\[
\\frac{}{} = \\frac{}{} \\frac{}{}, (11)
\\]

Now that \\( \\frac{\\partial z}{\\partial \\mathbf{x}^{i+1}} \\) is already computed and stored in memory.
It requires just a matrix reshaping operation (\\( vec \\)) and an additional transpose operation to get \\( \\frac{\\partial z}{\\partial (vec(\\mathbf{x}^{i+1})^T)} \\), which is the first term in the right hand side (RHS) of both equations.
So long as we can compute \\( \\frac{\\partial z}{\\partial (vec(\\mathbf{x}^{i+1})^T)} \\) and \\( \\frac{\\partial z}{\\partial (vec(\\mathbf{x}^{i+1})^T)} \\), we can easily get what we want (the left hand side of both equations).

The computation of \\( \\frac{\\partial vec(\\mathbf{x}^{i+1})}{\\partial (vec(\\mathbf{w}^i)^T)} \\) and \\( \\frac{\\partial vec(\\mathbf{x}^{i+1})}{\\partial(vec(\\mathbf{x}^i)^T)} \\) is not difficult in most cases, because \\( \\mathbf{x}^i \\) is directly related to \\( \\mathbf{x}^{i+1} \\), through a function with parameters \\( \\mathbf{w}^i \\).
The details of these partial derivatives will be discussed in the following sections.

## 4 The convolution layer ##

Now that the CNN architecture is clear, we will discuss in details the different types of layers, starting from the convolution layer.

### 4.1 Input, output, Filters, and notations ###

Suppose we are considering the l-th layer, whose inputs form a 3D tensor xl
with xl 2 RHlWlDl
. Thus, we need a triplet index set (il; jl; dl) to pointing
to any specic element in xl. That is, the triplet (il; jl; dl) refers to one element
in xl, which is in the dl-th slice / channel, and at spatial location (il; jl) (at the
il-th row, and jl-th column). In actual CNN learning, a mini-batch strategy is
usually used. In that case, xl becomes a 4D tensor in RHlWlDlN where N
is the mini-batch size. For simplicity we assume that N = 1 in this note.
In order to simplify the notations which will appear later, we follow the
zero-based indexing convention, which species that 0  il < Hl, 0  jl < Wl,
and 0  dl < Dl.
