---
layout: default
permalink: /dkhrs/
---

# Door Knob Hand Recognition System #

Qu, Xiaofeng; Zhang, David; Lu, Guangming; and Guo, Zhenhua, "Door knob hand recognition system," *Will appear in Systems, Man, and Cybernetics: Systems, IEEE Transactions on*

## Abstract ##

Biometric applications have been used globally in everyday life. However, conventional biometrics is created and optimized for high security scenarios. Being used in daily life by ordinary untrained people is a new challenge. Facing this challenge, designing a biometric system with prior constraints of ergonomics, we propose ergonomic biometrics design model, which attains the physiological factors, the psychological factors, and the conventional security characteristics. With this model, a novel hand based biometric system, door knob hand recognition system, is proposed. Door knob hand recognition system has the identical appearance of a conventional door knob, which is an optimum solution in both physiological factors and psychological factors. In this system, a hand image is captured by door knob imaging scheme, which is a tailored omni-vision imaging structure and is optimized for this predetermined door knob appearance. Then features are extracted by local Gabor binary pattern histogram sequence method and classified by projective dictionary pair learning. In the experiment on a large data set including 12000 images from 200 people, The proposed system achieves competitive recognition performance comparing with conventional biometrics like face and fingerprint recognition systems, with an equal error rate of 0.091%. This study shows that a biometric system could be built with a reliable recognition performance under the ergonomic constraints.

## 1 Introduction ##

**In** the last decade, biometrics has expended dramatically and globally. Biometrics came under the spotlight after the counter-terrorism war began. It has become a major solution for identity recognition and authentication. Since 2006, America has been requiring biometric passports for travelers entering the United States under the visa waiver program [[1][ref1]]. Not only in America, biometric technologies have been applied in identity documents in fifteen countries [[2][ref2], [3][ref3], [4][ref4]].

With the wide spread of biometrics, it has been used not only in boarder control, forensics and law enforcement agencies [[5][ref5]], but also in everyday life, such as, in smart cars [[6][ref6]] and smart homes [[7][ref7]]. Applications like these have educated the general public about the convenience and high-security level of biometric systems. People have accepted biometric systems as a common, convenient, and secure access control solution. Therefore, though biometrics has been designed to be used in high-security applications, there has been a great demand of biometrics in everyday life.

However, biometrics has not been designed in a user-friendly way. In current biometrics design, ergonomics (human factors) has not been a priority target. In the biometric system design [[8][ref8]], ergonomic study was limited. When designing a biometric system, only seven elements have been addressed: universality, distinctiveness, permanence, collectability, performance, acceptability, and circumvention [[5][ref5], [9][ref9]]. When testing a biometric system, only recognition related performance has been considered [[10][ref10]]. Ergonomics, to some extent, has been neglected.

In this paper, we present ergonomic biometrics design (EBD) model that considers ergonomics in all aspects of the design (in [Section 2](#section2)). We propose a door knob hand recognition system (DKHRS), which is shaped like a standard door knob, but incorporates a customized imaging device, a robust feature extraction, and a discriminative classification method (in Section 3). When addressing the imaging problem of this new device in this space-limited and shape-confined case, we propose a simplified catadioptric imaging structure - door knob imaging (DKI) scheme (in Section 3.2). The DKI scheme captures the surrounding hand skin surface in one omni-vision image in a cost-efficient structure. In the proposed system, we employ a local Gabor binary pattern histogram sequence (LGBPHS) method, which extracts robust histograms of dense local feature from DKHRS images (in Section 3.3.2). The features are classified using the dictionaries learned by projective dictionary pair learning (DPL) (in Section 3.3.3). Combining DKI scheme, LGBPHS method, and DPL method, we make the proposed DKHRS effective and efficient under the ergonomic constraints. The DKHRS has been used to collect an experimental data set of a significant scale. The experiment result on this data set is promising (in Section 4). The conclusion and future work is summarized in Section 5.

<a name="section2">
## 2 Ergonomic Biometrics Design ##
</a>

### 2.1 Development of Biometric Systems ###

Biometric systems are identity authentication systems utilizing various biological and/or behavioral traits, including fingerprint [[11][ref11]], face [[12][ref12]], hand/finger geometry, iris [[13][ref13], [14][ref14], [15][ref15]], sclera [[16][ref16], [17][ref17]], signature, gait, palmprint [[18][ref18], [19][ref19]], voice pattern, ear [[20][ref20]], hand vein, odor, and the DNA information of a person. A biometric system includes a sample collecting module (device), a feature extraction module, a database module, and a classification module [[21][ref21]]. a biometric system provides verification and/or identification functions [[21][ref21]]. The design cycle of a biometric system includes understanding the nature of the application, choosing biometrics trait, collecting biometric data, choosing features and classification algorithm and evaluating the system. During the development of a biometric system, seven factors: universality, distinctiveness, permanence, collectability, performance, acceptability, and circumvention are considered important [[5][ref5], [9][ref9], [21][ref21]]. Another influential study summarizes that a biometric system should be assessed by attributes including distinctiveness, stability, scalability, usability, inclusiveness, insensitivity, vulnerability, privacy, maintenance, health, quality, integration and cost [[8][ref8]].

### 2.2 Ergonomics Studies in Biometrics ###

From the above studies, it is noticed that ergonomics takes a significant role in developing biometric systems. Ergonomics (Human Factors) is to design products and systems in considering the interaction with people [[22][ref22]]. In biometrics, physical ergonomics and cognitive ergonomics are two critical factors. Physical ergonomics focuses on physical motion related traits including human anatomical, anthropometric, physiological and biomechanical characteristics. Cognitive ergonomics focuses on human-system interaction related mental activities including perception, memory, reasoning, and motor response. In biometric systems, ergonomics presents various significant factors. The Collectability/Health might be categorized to physical ergonomics. The Acceptability/Usability might fall into the area of cognitive ergonomics. In biometric systems, the ideal ergonomic solution would make the user barely notice the authentication process [[8][ref8]]. Also, a biometric system with poor ergonomics would jeopardize the quality of collected biometric samples [[23][ref23]]. For example, iris systems require fixed height of eyes. When using these systems, tall or short people encounter frustrations [[8][ref8]]. Another example is that small fingerprint sensors without guides capture fingerprints of poor quality [[8][ref8]].

Early ergonomics studies in development of biometric systems aimed at user acceptance [[24][ref24], [25][ref25], [26][ref26]], latent fingerprint examination [[27][ref27], [28][ref28]], and collectability  [[5][ref5], [8][ref8], [21][ref21], [29][ref29], [30][ref30]]. According to Albrecht [[24][ref24]], users accept natural and everyday motions the most readily, which conforms with ergonomic principles both in physical and cognitive. The analysis of latent fingerprint is widely depended on the human judgments because the analysis is semi-automatic [[27][ref27], [28][ref28]]. In collectability, the physical ergonomics is the challenge from an engineering perspective to next generation biometrics [[5][ref5], [30][ref30]]. Then, addressing ergonomic issues in a systematic view, human-biometric sensor interaction (HBSI) model is proposed [[31][ref31], [32][ref32]].

<a name="fig_hbsi">
![HBSI model](/images/fig_hbsi.svg)
</a>
*Fig. 1 Human-Biometric Sensor Interaction (HBSI) model shows the interactions between human, biometric sensors, and biometric systems (Redrawn in accordance with [[33][ref33]]).*

HBSI model, which is illustrated in [Fig. 1](#fig_hbsi), focuses on the interactions between target subjects and the biometric sensors [[31][ref31], [32][ref32], [34][ref34]]. HBSI model utilizes the metrics from both biometrics and ergonomics to assess the functionality and performance of biometric systems. Human-sensor intersection focuses on the physical ergonomics. Human-biometric system intersection represents the interactions between users and systems, which include sensors, software and implementations of systems. The aim of this intersection is comprised of three factors: effectiveness, efficiency, and satisfaction. Sensor-biometric system intersection address the image/sample quality issue.

HBSI model has been used to examine the ergonomics of swipe fingerprint sensors [[31][ref31], [34][ref34], [35][ref35]], and hand geometry machine [[36][ref36]]. HBSI model provides an adequate and thorough evaluation framework for biometric systems. Under the assessment of HBSI model, the functionality and performance of a biometric system can be characterized. However, HBSI has several restrictions. First, HBSI model, to a large extent, is an evaluation model. It contributes significantly in assessing a variety of biometric systems, but not in crafting new biometrics. Second, HBSI model evaluates interactions in the sensor level. In HBSI evaluations, either fingerprint sensors or hand geometry machines are taken as the elementary subject. The appearance and structure of the sensor constrained the enhancement of ergonomics in biometric systems. Third, in HBSI model, the ergonomics should be addressed at feature level. It is the biometric feature which defines the interactions among human, biometric sensors and biometric systems. The biometric feature determines the structure of sensors. For example, fingerprints are captured by semiconductor swipe fingerprint sensors or optical fingerprint sensors; hand geometries are obtained by hand geometry image capturing devices. In addition, the biometric feature defines a large part of interactions. For example, the iris system requires users to stand straight in front of the camera; the swipe fingerprint sensors require users to swipe with fingers. In order to enhance the functionality and the performance of biometric systems, ergonomics should be addressed at feature level.

### 2.3 Ergonomic Biometrics Design Model ###

EBD model is proposed to address the above limitations. EBD model implements four concepts. 

+ Considering ergonomics in the first stage - selecting biological and behavioral characteristics.
+ Considering ergonomics in all developing stages - selecting biological and behavioral characteristics, designing the sample-collecting device and designing the feature extraction and classification method.
+ Considering both physical and cognitive ergonomics in each stage.
+ Collaborating the recognition performance with ergonomics.

EBD model is illustrated in [Fig. 2](#fig_newmodel). This model is to provide a guideline for creating a new ergonomic biometric system. In design theory, Rubin and Chisnell [[37][ref37]] argue that there are five reasons why a machine or a system is difficult to use. The fundamental issue is that the focus has been on the machine or the system and not on users during the development. In a biometric system, the emphasis should be users. The consideration of ergonomics should be as early as in the first design stage and also in the full design process. Conventionally, the ergonomics analysis is occurred in the implementation stage, which is after the production of sensors. When the feature and the sensor is determined, the room for ergonomics is limited. In EBD model, we insist that ergonomics should be included in the full biometrics system design process. In all three stages: selecting biological and behavioral characteristics, designing the sample-collecting device and designing the feature extraction and classification method, ergonomics should be considered. Furthermore, both categories of ergonomics should be considered including the physical ergonomics and the cognitive ergonomics. Though ergonomics is an essential element in EBD model, recognition performance should not be ignored. The ergonomics and recognition performance should collaborate with each other during the design process.

<a name="fig_newmodel">
![EBD model](/images/fig_newmodel.svg)
</a>
*Fig. 2 EBD model considers both physical ergonomics and cognitive ergonomics in all three stages of biometric system development.*













[ref1]: http://www.usembassy.org.uk/visaservices/?p=420
[ref2]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6650265
[ref3]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5212583
[ref4]: https://www.loc.gov/law/help/biometric-data-retention/biometric-passport-data-retention.pdf
[ref5]: http://link.springer.com/chapter/10.1007%2F978-94-007-3892-8_3
[ref6]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6158802
[ref7]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5637551
[ref8]: http://www.springer.com/br/book/9780387776255
[ref9]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1262027
[ref10]: http://www.idsysgroup.com/ftp/BestPractice.pdf
[ref11]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4717822
[ref12]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4717827
[ref13]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5438771
[ref14]: http://www.springer.com/gp/book/9781447144014
[ref15]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6473911
[ref16]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6065764
[ref17]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=7365460
[ref18]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1227981
[ref19]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1678026
[ref20]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5438734
[ref21]: http://www.springer.com/gp/book/9780387773254
[ref22]: http://www.springer.com/gp/book/9783642012921
[ref23]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6095497
[ref24]: http://www.sciencedirect.com/science/article/pii/S0969476501001242
[ref25]: http://www.sciencedirect.com/science/article/pii/S0969476505703696
[ref26]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4263251
[ref27]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5467207
[ref28]: http://www.nist.gov/manuscript-publication-search.cfm?pub_id=910745
[ref29]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=6041850
[ref30]: http://www.springer.com/gp/book/9789400738911
[ref31]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=4105321
[ref32]: http://link.springer.com/chapter/10.1007%2F978-3-540-73321-8_102
[ref33]: http://link.springer.com/chapter/10.1007%2F978-3-642-02559-4_19
[ref34]: http://docs.lib.purdue.edu/dissertations/AAI3337302/
[ref35]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5422748
[ref36]: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=5678710
[ref37]: http://as.wiley.com/WileyCDA/WileyTitle/productCd-0470185481.html