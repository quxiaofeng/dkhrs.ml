---
layout: default
---

[Door Knob Hand Recognition System]({{site.url}}/)
==================================

## [Xiaofeng QU][csxfqu] ##

[**English**](/) \\( \\qquad \\) [中文](/cn/)

![DKHRS](/images/fig_device.png)
Door Knob Hand Recognition System

![The prototype](/images/fig_prototype.png)
Door Knob Hand Recognition System Prototype

![Working Scenario](/images/fig_scenario.png)
Working Scenario

## Background ##

+ Biometrics has been utilized worldwide.
+ A great amount of users have been educated of the convenience and security of biometric systems.
+ There is an expanding need of biometric systems in everyday life by ordinary people.
+ However, the majority of the biometric systems are designed for professionals or experienced people and tend to consider ergonomics a secondary element in system developing.

## Ergonomic Biometrics Design Model ##

**Four Principles of EBD Model**

+ Considering ergonomics in the first stage --- selecting biological and behavioral characteristics.
+ Considering ergonomics in all developing stages --- selecting biological and behavioral characteristics, designing the sample-collecting device and designing the feature extraction and classification method.
+ Considering both physical and cognitive ergonomics in each stage.
+ Collaborating the recognition performance with ergonomics.

![EBD model](/images/fig_newmodel.svg)
EBD Model

## Door Knob Hand Recognition System ##

**"Reinvent the Door Knob"**, it is not a new biometrics, but is a new door knob.

![The basic idea](/images/fig_origin.png)
The basic idea

**"Open the door just like it is not locked."**

### Imaging ###

+ Conventional Imaging: small view field, large, easy to be interfered
+ Catadioptric Imaging: compact, capturing the surroundings in one image; large, expensive
+ Door Knob Imaging: low-cost, capturing the surroundings in one image, short working distance

![Conventional imaging scheme](/images/fig_conventionalimaging.svg)
Conventional imaging scheme

![Catadioptric imaging scheme](/images/fig_cata.png)
Catadioptric imaging scheme

![Door Knob Imaging scheme](/images/fig_doorknobimaging.png)
Door Knob Imaging scheme

## Feature Extraction and Classification ##

### Preprocessing ###

![Calibration](/images/fig_calibration.png)
System Calibration

![Ring area](/images/fig_ring.png)
The area of hand

### Feature Extraction ###

![LGBPHS Method](/images/fig_lgbphs.svg)
Local Gabor Binary Pattern Histogram Sequence

### Projective Dictionary Pair Learning

See [DPL](/dpl/) and [DPL Supplementary](/dpl-supplementary/).

### Performance ###

+ The best EER is 0.091%.
+ The recognition rate of DKHRS is over 99%, and its EER can be lower than 0.1%.
+ Generally speaking, the recognition performance of DKHRS is much better than hand back skin texture [Xie2012](#xie2012), gait [Lai2014gait](#lai2014gait) and face recognition [Gu2014dpl](#gu2014dpl), [Lfw2015](#lfw2015);
+ it is even surpass fingerprint recognition (about 1% EER on STFV-STD-1.0 dataset [ICB2013](#icb2013)) and 3D fingerprint (3.4% EER [Liu2015](#liu2015));
+ but it is still not as good as iris recognition (<0.003% EER[Daugman2007](#daugman2007)), and palmprint recognition (EER from 0.062% to 0.012% [Zuo2008compcode](#zuo2008compcode), [Guo2009bocv](#guo2009bocv), [Laadjel2009a](#laadjel2009a), [Guo2009coc](#guo2009coc), [Zhang2010](#zhang2010), [Zhang2010b](#zhang2010b), [Li2012a](#li2012a), [Qu2015lps](#qu2015lps)).

## Publications ##

### Paper ###

+ [Qu, Xiaofeng][csxfqu]; [Zhang, David][csdzhang]; [Lu, Guangming][csgmlu]; and [Guo, Zhenhua][cszhguo], "[Door knob hand recognition system][dkhrs]," *in Systems, Man, and Cybernetics: Systems, IEEE Transactions on , vol.PP, no.99, pp.1-12*.

### Patents ###

+ Door Knob Hand Image Accquisition Apparatus and Door Knob Hand Recognition System
  + Grant
  + [China Patent CN203786745 (U)](https://www.google.com/patents/CN203786745U?cl=en&dq=CN203786745)


+ Door Knob Hand Image Recognition System and Identification Method
  + Application
  + [China Patent CN104978551 (A)](http://www.google.com/patents/CN104978551A?cl=en)

[csxfqu]: http://www.quxiaofeng.me/about
[csdzhang]: http://www4.comp.polyu.edu.hk/~csdzhang/
[csgmlu]: http://www.hitsz.edu.cn/body/shizi/detailen.php?strID=396
[cszhguo]: http://www.sz.tsinghua.edu.cn/publish/sz/139/2012/20120420104947649501973/20120420104947649501973_.html
[dkhrs]: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7433472

## References ##

+ <a name="xie2012"></a>[J. Xie, L. Zhang, J. You, D. Zhang, and X. Qu, "A study of hand back skin texture patterns for personal identification and gender classification." Sensors (Basel, Switzerland), vol. 12, no. 7, pp. 8691-709, 1 2012.][Xie2012]
+ <a name="lai2014gait"></a>[Z. Lai, Y. Xu, Z. Jin, and D. Zhang, "Human gait recognition via sparse discriminant projection learning," IEEE Transactions on Circuits and Systems for Video Technology, vol. 24, no. 10, pp. 1651-1662, 10 2014.][Lai2014gait]
+ <a name="gu2014dpl"></a>[S. Gu, L. Zhang, W. Zuo, and X. Feng, "Projective dictionary pair learning for pattern classification," in Advances in Neural Information Processing Systems 27, Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, Eds. Curran Associates, Inc., 2014, pp. 793-801.][Gu2014dpl]
+ <a name="lfw2015"></a>["Lfw : Results." [Online]. Available: http://vis-www.cs.umass.edu/lfw/results.html][Lfw2015]
+ <a name="icb2013"></a>["Fvc-ongoing." [Online]. Available: https://biolab.csr.unibo.it/fvcongoing/UI/Form/ICB2013STFV.aspx][ICB2013]
+ <a name="liu2015"></a>[F. Liu, D. Zhang, and L. Shen, "Study on novel curvature features for 3d fingerprint recognition," Neurocomputing, vol. 168, pp. 599-608, 11 2015.][Liu2015]
+ <a name="daugman2007"></a>[J. Daugman, "New methods in iris recognition," Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on, vol. 37, no. 5, pp. 1167-1175, 10 2007.][Daugman2007]
+ <a name="zuo2008compcode"></a>[W. Zuo, F. Yue, K. Wang, and D. Zhang, "Multiscale competitive code for efficient palmprint recognition," in Proc. 19th Int. Conf. Pattern Recogn. (ICPR), 2008, pp. 1-4.][Zuo2008compcode]
+ <a name="guo2009bocv"></a>[Z. Guo, D. Zhang, L. Zhang, and W. Zuo, "Palmprint verification using binary orientation co-occurrence vector," Pattern Recogn. Lett., vol. 30, no. 13, pp. 1219-1227, 10 2009.][Guo2009bocv]
+ <a name="laadjel2009a"></a>[M. Laadjel, F. Kurugollu, A. Bouridane, and W. Yan, "Palmprint recognition based on subspace analysis of gabor filter bank," in Proc. 10th Pacific Rim Conf. Multimedia: Advances in Multimedia Information Processing (PCM). Springer-Verlag, 2009, pp. 719-730.][Laadjel2009a]
+ <a name="guo2009coc"></a>[Z. Guo, W. Zuo, L. Zhang, and D. Zhang, "Palmprint verification using consistent orientation coding," in Proc. 16th IEEE Int. Conf. Image Process. (ICIP), 2009, pp. 1965-1968.][Guo2009coc]
+ <a name="zhang2010"></a>[D. Zhang, Z. Guo, G. Lu, and W. Zuo, "An online system of multispectral palmprint verification," IEEE Trans. Instrum. Meas., vol. 59, no. 2, pp. 480-490, 2 2010.][Zhang2010]
+ <a name="zhang2010b"></a>[D. Zhang, V. Kanhangad, N. Luo, and A. Kumar, "Robust palmprint verification using 2d and 3d features," Pattern Recogn., vol. 43, no. 1, pp. 358-368, 1 2010.][Zhang2010b]
+ <a name="li2012a"></a>[W. Li, D. Zhang, G. Lu, and N. Luo, "A novel 3-d palmprint acquisition system," Systems, Man and Cybernetics, Part A: Systems and Humans, IEEE Transactions on, vol. 42, no. 2, pp. 443-452, 3 2012.][Li2012a]
+ <a name="qu2015lps"></a>[X. Qu, D. Zhang, and G. Lu, "A novel line-scan palmprint acquisition system," Systems, Man, and Cybernetics: Systems, IEEE Transactions on, vol. PP, pp. 1-11, 2016.][Qu2015lps]

[Xie2012]: http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3444070/
[Lai2014gait]: http://ieeexplore.ieee.org/xpl/articleDetails.jsp?tp=&arnumber=6737218
[Gu2014dpl]: https://papers.nips.cc/paper/5600-projective-dictionary-pair-learning-for-pattern-classification
[Lfw2015]: http://vis-www.cs.umass.edu/lfw/results.html
[ICB2013]: https://biolab.csr.unibo.it/fvcongoing/UI/Form/ICB2013STFV.aspx
[Liu2015]: http://www.sciencedirect.com/science/article/pii/S0925231215007638
[Daugman2007]: http://www.cl.cam.ac.uk/~jgd1000/NewMethodsInIrisRecog.pdf
[Zuo2008compcode]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4761868
[Guo2009bocv]: http://www4.comp.polyu.edu.hk/~cslzhang/paper/PRL_09_Oct.pdf
[Laadjel2009a]: http://link.springer.com/chapter/10.1007%2F978-3-642-10467-1_63
[Guo2009coc]: http://www4.comp.polyu.edu.hk/~cslzhang/paper/conf/ICIP09_Denis.pdf
[Zhang2010]: http://www4.comp.polyu.edu.hk/~biometrics/MultispectralPalmprint/An%20Online%20System%20of%20Multispectral%20Palmprint%20Verification.pdf
[Zhang2010b]: http://www4.comp.polyu.edu.hk/~csajaykr/myhome/papers/PR10.pdf
[Li2012a]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6017138
[Qu2015lps]: http://www.doorknob.ml