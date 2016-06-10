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