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
   2. The forward run
   3. Stochastic gradient descent (SGD)
   4. Error back propagation
4. The convolution layer
   1. Input, output, filters, and notations
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
\\mathbf{x}^1 \\rightarrow \[\\mathbf{w}^1\] \\rightarrow \\mathbf{x}^2 \\rightarrow \\ldots \\rightarrow \\mathbf{x}^{L-1} \\rightarrow \[\\mathbf{w}^{L-1}\] \\rightarrow \\mathbf{x}^L \\rightarrow \[\\mathbf{w}^L\] \\rightarrow \\mathbf{z} \\qquad (5) 
\\]

The above [Equation 5](#eqn_abstract_cnn_structure) illustrates how a CNN runs layer by layer forward.
The input is \\( \\mathbf{x}^1 \\), usually an image (3D tensor). It goes through the processing in the first layer, which is the first box.
We denote the parameters involved in the first layer's processing collectively as a tensor \\( \\mathbf{w}^1 \\).
The output of the first layer is \\( \\mathbf{x}^2 \\), which also acts as the input to the second layer processing.

This processing proceeds till all layers in the CNN has been finished, which outputs \\( \\mathbf{x}^L \\).
One additional layer, however, is added for backward error propagation.
Let’s suppose the problem at hand is an image classification problem with \\( C \\) classes.
A commonly used strategy is to output \\( \\mathbf{x}^L \\) as a \\( C \\) dimensional vector, whose \\( i \\)-th entry encodes the prediction \\( P(c\_i \\mid \\mathbf{x}^1 \\).
That is, given the input \\( \\mathbf{x}^1 \\), \\( \\mathbf{x}^L \\) outputs a vector of CNN-estimated posterior probabilities.

Let's suppose \\( \\mathbf{t} \\) is the corresponding target (ground-truth) value for the input
\\( \\mathbf{x}^1 \\), then a cost or loss function can be used to measure the discrepancy between the CNN prediction \\( \\mathbf{x}^L \\) and the target \\( \\mathbf{t} \\).
For example, the simplest loss function could be
<a name="eqn_simplest_loss"></a>\\[
\\qquad \\qquad \\qquad z = \\| \\mathbf{t} - \\mathbf{x}^L \\|^2, \\qquad \\qquad (6)
\\]
although more complex loss functions are usually used.

For simplicity, [Equation 6](#eqn-simplest-loss) explicitly models the loss function as a loss layer, whose processing is modeled as a box with parameters \\( \\mathbf{w}^L \\).

Note that some layers may not have any parameters, that is, \\( \\mathbf{w}^i \\) may be empty for some \\( i \\).