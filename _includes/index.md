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

Discriminative dictionary learning (DL) has been widely studied in various pattern classification problems. Most of the existing DL methods aim to learn a synthesis dictionary to represent the input signal while enforcing the representation coefficients and/or representation residual to be discriminative. However, the \\(l\_0\\) or \\(l\_1\\)-norm sparsity constraint on the representation coefficients adopted in most DL methods makes the training and testing phases time consuming. We propose a new discriminative DL framework, namely projective dictionary pair learning (DPL), which learns a synthesis dictionary and an analysis dictionary jointly to achieve the goal of signal representation and discrimination. Compared with conventional DL methods, the proposed DPL method can not only greatly reduce the time complexity in the training and testing phases, but also lead to very competitive accuracies in a variety of visual classification tasks.

#### Introduction

Sparse representation represents a signal as the linear combination of a small number of atoms chosen out of a dictionary, and it has achieved a big success in various image processing and computer vision applications [1, 2]. The dictionary plays an important role in the signal representation process [3]. By using a predefined analytical dictionary (e.g., wavelet dictionary, Gabor dictionary) to represent a signal, the representation coefficients can be produced by simple inner product operations. Such a fast and explicit coding makes analytical dictionary very attractive in image representation; however, it is less effective to model the complex local structures of natural images.

Sparse representation with a synthesis dictionary has been widely studied in recent years [2, 4, 5]. With synthesis dictionary, the representation coefficients of a signal are usually obtained via an \\(l\_p\\)-norm (\\(p\\leq1\\)) sparse coding process, which is computationally more expensive than analytical dictionary based representation. However, synthesis based sparse representation can better model the complex image local structures and it has led to many state-of-the-art results in image restoration [6]. Another important advantage lies in that the synthesis based sparse representation model allows us to easily learn a desired dictionary from the training data. The seminal work of KSVD [1] tells us that an over-complete dictionary can be learned from example natural images, and it can lead to much better image reconstruction results than the analytically designed off-the-shelf dictionaries. Inspired by KSVD, many dictionary learning (DL) methods have been proposed and achieved state-of-the-art performance in image restoration tasks.

The success of DL in image restoration problems triggers its applications in image classification tasks. Different from image restoration, assigning the correct class label to the test sample is the goal of classification problems; therefore, the discrimination capability of the learned dictionary is of the major concern. To this end, supervised dictionary learning methods have been proposed to promote the discriminative power of the learned dictionary [4, 5, 7, 8, 9]. By encoding the query sample over the learned dictionary, both the coding coefficients and the coding residual can be used for classification, depending on the employed DL model. Discriminative DL has led to many state-of-the-art results in pattern recognition problems.

One popular strategy of discriminative DL is to learn a shared dictionary for all classes while enforcing the coding coefficients to be discriminative [4, 5, 7]. A classifier on the coding coefficients can be trained simultaneously to perform classification. Mairal et al. [7] proposed to learn a dictionary and a corresponding linear classifier in the coding vector space. In the label consistent KSVD (LC-KSVD) method, Jiang et al. [5] introduced a binary class label sparse code matrix to encourage samples from the same class to have similar sparse codes. In [4], Mairal et al. proposed a task driven dictionary learning (TDDL) framework, which minimizes different risk functions of the coding coefficients for different tasks.

Another popular line of research in DL attempts to learn a structured dictionary to promote discrimination between classes [2, 8, 9, 10]. The atoms in the structured dictionary have class labels, and the class-specific representation residual can be computed for classification. Ramirez et al. [8] introduced an incoherence promotion term to encourage the sub-dictionaries of different classes to be independent. Yang et al. [9] proposed a Fisher discrimination dictionary learning (FDDL) method which applies the Fisher criterion to both representation residual and representation coefficient. Wang et al. [10] proposed a max-margin dictionary learning (MMDL) algorithm from the large margin perspective.

In most of the existing DL methods, \\(l\_0\\)-norm or \\(l\_1\\)-norm is used to regularize the representation coefficients since sparser coefficients are more likely to produce better classification results. Hence a sparse coding step is generally involved in the iterative DL process. Although numerous algorithms have been proposed to improve the efficiency of sparse coding [11, 12], the use of \\(l\_0\\)-norm or \\(l\_1\\)-norm sparsity regularization is still a big computation burden and makes the training and testing inefficient.

It is interesting to investigate whether we can learn discriminative dictionaries but without the costly \\(l\_0\\)-norm or \\(l\_1\\)-norm sparsity regularization. In particular, it would be very attractive if the representation coefficients can be obtained by linear projection instead of nonlinear sparse coding. To this end, in this paper we propose a projective dictionary pair learning (DPL) framework to learn a synthesis dictionary and an analysis dictionary jointly for pattern classification. The analysis dictionary is trained to generate discriminative codes by efficient linear projection, while the synthesis dictionary is trained to achieve class-specific discriminative reconstruction. The idea of using functions to predict the representation coefficients is not new, and fast approximate sparse coding methods have been proposed to train nonlinear functions to generate sparse codes [13, 14]. However, there are clear difference between our DPL model and these methods. First, in DPL the synthesis dictionary and analysis dictionary are trained jointly, which ensures that the representation coefficients can be approximated by a simple linear projection function. Second, DPL utilizes class label information and promotes discriminative power of the representation codes.

One related work to this paper is the analysis-based sparse representation prior learning [15, 16], which represents a signal from a dual viewpoint of the commonly used synthesis model. Analysis prior learning tries to learn a group of analysis operators which have sparse responses to the latent clean signal. Sprechmann et al. [17] proposed to train a group of analysis operators for classification; however, in the testing phase a costly sparsity-constrained optimization problem is still required. Feng et al. [18] jointly trained a dimensionality reduction transform and a dictionary for face recognition. The discriminative dictionary is trained in the transformed space, and sparse coding is needed in both the training and testing phases.

The contribution of our work is two-fold. First, we introduce a new DL framework, which extends the conventional discriminative synthesis dictionary learning to discriminative synthesis and analysis dictionary pair learning (DPL). Second, the DPL utilizes an analytical coding mechanism and it largely improves the efficiency in both the training and testing phases. Our experiments in various visual classification datasets show that DPL achieves very competitive accuracy with state-of-the-art DL algorithms, while it is significantly faster in both training and testing.

#### Projective Dictionary Pair Learning

Denote by \\(\\mathbf{X = \[X\_1,\\ldots,X\_k,\\ldots,X\_K\]}\\) a set of p-dimensional training samples from \\(K\\) classes, where \\(\\mathbf{X\_k} \\in \\Re^{p\\times{}n}\\) is the training sample set of class \\(k\\), and \\(n\\) is the number of samples of each class. Discriminative DL methods aim to learn an effective data representation model from \\(\\mathbf{X}\\) for classification tasks by exploiting the class label information of training data. Most of the state-of-the-art discriminative DL methods [5, 7, 9] can be formulated under the following framework:

\\[\min\_{\\mathbf{D,A}}\\|\\mathbf{X-DA}\\|^2\_F+\\lambda\\|\\mathbf{A}\\|\_p+\\mathbf{\\Psi(D,A,Y)},\\]

where \\(\\lambda \geq 0\\) is a scalar constant, \\(\\mathbf{Y}\\) represents the class label matrix of samples in \\(\\mathbf{X}\\), \\(\\mathbf{D}\\) is the synthesis dictionary to be learned, and \\(\\mathbf{A}\\) is the coding coefficient matrix of \\(\\mathbf{X}\\) over \\(\\mathbf{D}\\). In the training model above, the data fidelity term \\(\\|\\mathbf{X-DA}\\|^2\_F\\) ensures the representation ability of \\(\\mathbf{D}\\); \\(\\|\\mathbf{A}\\|\_p\\) is the \\(l\_p\\)-norm regularizer on \\(\\mathbf{A}\\); and \\(\\mathbf{\\Psi(D,A,Y)}\\) stands for some discrimination promotion function, which ensures the discrimination power of \\(\\mathbf{D}\\) and \\(\\mathbf{A}\\).

As we introduced in last Section, some DL methods [4, 5, 7] learn a shared dictionary for all classes and a classifier on the coding coefficients simultaneously, while some DL methods [8, 9, 10] learn a structured dictionary to promote discrimination between classes. However, they all employ \\(l\_0\\) or \\(l\_1\\)-norm sparsity regularizer on the coding coefficients, making the training stage and the consequent testing stage inefficient.

In this work, we extend the conventional DL model, which learns a discriminative synthesis dictionary, to a novel DPL model, which learns a pair of synthesis and analysis dictionaries. No costly \\(l\_0\\) or \\(l\_1\\)-norm sparsity regularizer is required in the proposed DPL model, and the coding coefficients can be explicitly obtained by linear projection. Fortunately, DPL does not sacrifice the classification accuracy while achieving significant improvement in the efficiency.

##### Discriminative dictionary learning

The conventional discriminative DL model in (1) aims to learn a synthesis dictionary \\(\\mathbf{D}\\) to sparsely represent the signal \\(\\mathbf{X}\\), and a costly \\(l_1\\)-norm sparse coding process is needed to resolve the code \\(\\mathbf{A}\\). Suppose that if we can find an analysis dictionary, denoted by \\(\\mathbf{P}\\in\\Re^{mK\\times{}p}\\), such that the code \\(\\mathbf{A}\\) can be analytically obtained as \\(\\mathbf{A=PX}\\), then the representation of \\(\\mathbf{X}\\) would become very efficient. Based on this idea, we propose to learn such an analysis dictionary \\(\\mathbf{P}\\) together with the synthesis dictionary \\(\\mathbf{D}\\), leading to the following DPL model:

\\[\\{\\mathbf{P^*,D^*}\\}={argmin}\_{P,D}\\|X-DPX\\|^2\_F+\\Psi(D,P,X,Y),\\]

where 	(D; P;X; Y) is some discrimination function. D and P form a dictionary pair: the analysis dictionary P is used to analytically code X, and the synthesis dictionary D is used to reconstruct X. The discrimination power of the DPL model depends on the suitable design of 	(D; P;X; Y) . We propose to learn a structured synthesis dictionary D = [D1; : : : ;Dk; : : : ;DK] and a structured  forms a subdictionary pair corresponding to class k. Recent studies on sparse subspace clustering [19] have proved that a sample can be represented by its corresponding dictionary if the signals satisfy certain incoherence condition. With the structured analysis dictionary P, we want that the sub-dictionary Pk can project the samples from class i; i 6= k, to a nearly null space, i.e.,



Clearly, with (3) the coefficient matrix PX will be nearly block diagonal. On the other hand, with the structured synthesis dictionary D, we want that the sub-dictionary Dk can well reconstruct the data matrix Xk from its projective code matrix PkXk; that is, the dictionary pair should minimize
the reconstruction error:


Based on the above analysis, we can readily have the following DPL model:

##### The dictionary pair learning model

##### Optimization

##### Classification scheme 

##### Complexity and convergence

### Experiment ###

Flowchart

![Flowchart](/images/fig_flowchart.svg)

## Publications ##

### Patents ###

### Papers ###