[中文版 (Chinese version)](/cn/)

![DKHRS](/images/fig_device.png)

## Ergonomic Biometrics Design Model ##

HBSI Model

![HBSI model](/images/fig_hbsi.svg)

EBD Model

![EBD model](/images/fig_newmodel.svg)

## Door Knob Hand Recognition System ##

### Concept and Framework ###

### Imaging ###

## Feature Extraction and Classification ##

LGBPHS

![LGBPHS Method](/images/fig_lgbphs.svg)

### Projective Dictionary Pair Learning

#### Abstract

Discriminative dictionary learning (DL) has been widely studied in various pattern classification problems. Most of the existing DL methods aim to learn a synthesis dictionary to represent the input signal while enforcing the representation coefficients and/or representation residual to be discriminative. However, the \\(l_0\\) or \\(l_1\\)-norm sparsity constraint on the representation coefficients adopted in most DL methods makes the training and testing phases time consuming. We propose a new discriminative DL framework, namely projective dictionary pair learning (DPL), which learns a synthesis dictionary and an analysis dictionary jointly to achieve the goal of signal representation and discrimination. Compared with conventional DL methods, the proposed DPL method can not only greatly reduce the time complexity in the training and testing phases, but also lead to very competitive accuracies in a variety of visual classification tasks.

#### Introduction

Sparse representation represents a signal as the linear combination of a small number of atoms chosen out of a dictionary, and it has achieved a big success in various image processing and computer vision applications [1, 2]. The dictionary plays an important role in the signal representation process [3]. By using a predefined analytical dictionary (e.g., wavelet dictionary, Gabor dictionary) to represent a signal, the representation coefficients can be produced by simple inner product operations. Such a fast and explicit coding makes analytical dictionary very attractive in image representation; however, it is less effective to model the complex local structures of natural images.

Sparse representation with a synthesis dictionary has been widely studied in recent years [2, 4, 5]. With synthesis dictionary, the representation coefficients of a signal are usually obtained via an \\(l_p\\)-norm (\\(p\\leq1\\)) sparse coding process, which is computationally more expensive than analytical dictionary based representation. However, synthesis based sparse representation can better model the complex image local structures and it has led to many state-of-the-art results in image restoration [6]. Another important advantage lies in that the synthesis based sparse representation model allows us to easily learn a desired dictionary from the training data. The seminal work of KSVD [1] tells us that an over-complete dictionary can be learned from example natural images, and it can lead to much better image reconstruction results than the analytically designed off-the-shelf dictionaries. Inspired by KSVD, many dictionary learning (DL) methods have been proposed and achieved state-of-the-art performance in image restoration tasks.

The success of DL in image restoration problems triggers its applications in image classification tasks. Different from image restoration, assigning the correct class label to the test sample is the goal of classification problems; therefore, the discrimination capability of the learned dictionary is

### Experiment ###

Flowchart

![Flowchart](/images/fig_flowchart.svg)

## Publications ##

### Patents ###

### Papers ###