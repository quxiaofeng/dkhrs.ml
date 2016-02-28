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

1. Motivation
2. Preliminaries
   1. Tensor and vectorization
   2. Vector calculus and chain rule
3. CNN in a nutshell
   1. The architecture
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
We use a symbol shown in boldface to represent a vector, e.g., \\( \\mathbf{x} \\in \\mathcal{R}^D \\) is a column vector with \\( D \\) components.
We use a capital letter to denote a matrix, e.g., \\( X \\in \\mathcal{R}^{H\\times{}W} \\) is a matrix with \\( H \\) rows and \\( H \\) columns. The vector \\( \\mathbf{x} \\) can also be viewed as a matrix with \\( 1 \\) column and \\( D \\) rows.

These concepts can be generalized to higher-order matrices, i.e., tensors.
For example, \\( \\mathbf{x} \\in \\mathcal{R}^{H\\times{}W\\times{}D} \\) is a 3D tensor.
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

\\[
A = \\begin{matrix}
1 & 2 \\\
3 & 4
\\end{matrix},
A(:) = (1,3,2,4)^T = \\begin{matrix}
1 \\\
2 \\\
3 \\\
4
\\end{matrix} \\qquad (1)
\\]