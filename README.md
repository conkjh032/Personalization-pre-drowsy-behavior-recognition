# Personalization-pre-drowsy-behavior-recognition

운전중에 졸음이 올 때, 특정한 행동을 하는 것을 관찰하게 되었습니다. 이것을 '졸음을 깨기 위한 행동'이라고 하겠습니다. 
이 프로젝트는 '졸음을 깨기 위한 행동'을 자동으로 학습하여 각각의 사용자에 맞는 모델이 만들어지도록 하는 알고리즘입니다.

<img src="https://ibb.co/JFm0Pc1" width="300" height="300"> <img src="https://ibb.co/CJQTFRH" width="300" height="300"> 

프로그램이 작동되는 흐름도는 [여기](https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&layers=1&nav=1&title=Pre-drowsy%20behavior%20detection#R7V1Zl6I4FP41eaw%2BQBIMj4DaPT09vZzeph9R08oUSg1Si%2FPrJwlhDaW4lGDJU5FLEjB3%2B3LvDQWgu3x6G3l3i7%2FCGQ2Aoc2eABwCwzAQhuwPp2wSyoBIwjzyZwlJzwlf%2Ff%2BoJGqSeu%2FP6LrUMQ7DIPbvysRpuFrRaVyieVEUPpa7%2FQ6D8lPvvDlVCF%2BnXqBSf%2FqzeJFQiTHI6e%2BoP1%2BkT9ZNK7kz8aa38yi8X8nnAQPqOnLwOLm99NK55A9dL7xZ%2BFggwRGAbhSGcXK1fHJpwNc2Xbaff2x%2BBh9uzbfvv6z%2F9b47f377%2BOMmmWy8z5DsF0Z0FR88NRy%2F%2Fz55CL88zj95P8aL0Jp8%2F3Sjy7VYx5t0PemMLa9shlG8COfhygtGOdURa0b5tBpr5X0%2BhOEdI%2BqM%2BA%2BN442UFe8%2BDhlpES8DeZc%2B%2BfHffPgbLFu%2FCneGT3Jm0dikjVUcbQqDePNX8V4%2BTLTScYE3oYGTsdoNgzASPzRlNnTWcRTe0sIdzSCaxQf%2FDldxgT5C4%2BGIv2ayaHylKjK4g0Oy3zq8j6Z0Sz%2BpgbEXzem2%2BcxMDJl603BJ2S9n4yIaeLH%2FUH45T%2BrZPOuXCwu7kPKyh1jKl3zwgnv5JDAaAgsBgsDIAbYOLBeMbOAMgEM4xRoBm7ARYOQCwjog0d8ERBXCXMT4cj8u%2FJh%2BvfPEmj0yM1YWJ84lKWs6Ww5nHnjrtWR%2FwtvUMPDemRoLBvtBUGDwGGGHmYfDRaKpBDzQKKZPBf6oTEzvmtL8SPOsY9l%2BzI2dnpqoRcHQIe14vtfbDF1hV28z2rcZpmozatlHjrQZYqgdRd6m0OEu9FfxujDzZ07IpThzolKKIa44q0p%2FXdvan10kb5BLcfZTjhBs4wIFGyKzKNo32hvNMHfIt2h9ppHPVoxG1yD0g0sQemgYLQj9oE2h1wsinyvATqE3MemF%2FsWF%2Fih0aKro0DAD9rrOJGJXc34l8B8GliUQIUOKWMBEjQNHDhMZRrQ5ZGQXZChApAssdoHqpnI54mS3eB8GNN108mS4IS7YLZsjVDhWEeciXE7u17vRZhk%2FMhEYe0s%2F4FL2jgYPNPanXg0m9QJ%2FvmKNKRMBLn11wJQ90l%2FNWcvMW9%2BEQt2gC8GquGTAMgNYxKp6DVbN3P3JwSq8RJ%2F%2BqsDq86apgQl7TuBObsO2vWXBho02lBGcwF%2FdqjZoyBzLNGZredxu9sRqrqhvUzY%2Bq%2BbsuWWcQhpuSc2X0nKjVRCzj5ZfpLaSptqq1cvNebSVKNr6OaI3Mx5h5i%2Fh0IX34HPtvFa9RV1TW10Fib%2FoWmEDW4a4vNYRXfv%2FeRPRga8k1821RFt1YIuvJcNlgS1vLP3ZTNgAuYVjj8EOwMMKG1fhiioclMSy9Ti7CTbKSAvWIC1Yw0rjxVipat%2FHsOdkE2dawcw18d2zctLoU0Jd9MEpv3c7YdimE0akVek5JJykl2QnF6WrlB50rqDR1tcs5RRd4GgyCmSNRbJQZBBtTSQaXWCJkFF9RMgARMSUbAhIHxFqAXKWnRuqg5xnDghpCqM76t4u09AcG7Bplkoxq9ntxLDJURUROUGWBLUqNYdlSa7BraUFZ7vcGjw2632cW0sj6b1bex1urbpp64Jfg72F6qKFqinlq%2B%2BI2rRQEF8KKrquTX9j6cGt%2Bre6UlAXEOadiCwFtYeiFNTg%2FipxdI7JHRQnMo%2BnFRL%2FqsdL0v9anv5nHo85SU5JhydzWiP5xBO6we57Ix0ab8r%2BKIPHRX80wDX%2BCL5YGBHVyETj8uAKwnnliRlYqfE1U1a1lpmBF5NQvS5%2FgJv6g3OdDNj6miXdV4q5uKYr9V9S%2FQfCSpzmdEDn1b9a4k%2F0thOzRk31Xm%2B8m3HP0lrnnpqL7dPqzXipl3mJieqIz5qMhWrkqGdlM0yldY2VKiC%2BzAoJJS54VrbqlZ1OTeTtvGx9NTVMrfK1WtA0aJuvyFB42NF46kXuZGDTmlLUagU46lBcVNsqBVdVztJYelqtZoEqDn%2F%2B4FJ6uInn89jFQCT%2F0pAnj2UmW%2BX0YBQhItLJdmCW2Io5%2FILNI4OgtSekRKzUyjbYWcyVyCeePHRa8ihqHrD7eztU2dshsyYwh%2BvyfC8WV8XmpVsk41WaJHwRJgm3WtzbZ4m3gpwGwoNbrWNBajRCZvWcJL0HeflK5rUSV8YTdTrIc37CWbFutgqt9ooMXrxzMQeVCEVN2Nc654ddkJrG7SMUB5yy2s3W8%2B5k1cBTH6E4QF2rseHW%2BapGnnp13ZutuHIqEsGW2Yrb%2FbzWIRDtGkIOqOknC3Crh6Bxq2WgB20PrwLgN5WedvG9%2BsGLZ8JTov7O0GQAyk4iUYTHlyT6d9MCDvxMTV9a28EjTmNZAiK%2F7aP1oar9%2FdiNPuhcrApZPUI5nrNdw524Q7WBh7mYVxqBbHrGt1Ufk75l1cdA4Lh5tqOPITW0%2BtmHP7oSRMKGwt8%2B2nAAY7tn9vv6pZPANNi1eIOapU5OpyKOsmW2OUPoOCUK0J18bjM%2FsjoSmD07k9Mw1%2F1imwk5rfgtdvIsAoheODBbzlnYyUmk5AOi1%2B5aSNWztF7XbKrQQS9L6pCfmiaot0p78npQ%2BXBrnbOp2zoeYJRYM%2F%2BXKcnnGfL%2FSwNH%2FwM%3D)에 올려두었습니다.


&nbsp;
&nbsp;
&nbsp;

## Table of contents

1. [Documentation](#Documentation)
2. [Development](#Development)
   * [Environment](#Environment)
   * [Techs](#Techs)
   * [Run](#Run)
   * [Performance](#Performance)
3. [Test](#Test)
4. [Reference](#Reference)
5. [Question Issues](#Question_Issues)
6. [Social Media](#Social)

&nbsp;
&nbsp;
&nbsp;

## Documentation


개발의 전반적인 과정은 [블로그](https://conkjh032.tistory.com/)에 올려놓을 계획입니다.

&nbsp;
&nbsp;
&nbsp;

## Development

### Environment :gear:

|Tool|OS|Pycharm|Python|Tensorflow|Pytorch|CUDA|cuDNN|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Version|Windows|2019.2.6|3.7.6|1.14.0|1.4.0|10.1|7.0|

&nbsp;
&nbsp;

### Technologies :pencil2:

이 프로젝트를 진행하기 위해서 필요한 기술들을 찾아서 사용했습니다. 사용된 기술들과 소개를 간략하게 하겠습니다.

&nbsp;

**1. Facial Landmarks**\
얼굴인식 분야에서 facial landmarks는 굉장히 중요한 기술입니다. Facial landmarks는 얼굴에 점들을 표시하고, 표시된 점들이 모여서 얼굴의 특징을 나타냅니다.

&nbsp;

**2. Eye Blink Detection**\
Facial Landmarks로 오른쪽 눈과 왼쪽 눈을 찾아냅니다. 그리고 눈꺼풀 주변 사이의 거리를 측정하여 눈이 감겼는지 감지합니다.

&nbsp;

**3. Siamese Network**\
Siamese Network는 다루어야 하는 클래스의 종류는 많지만, 그에 비해 데이터를 많이 구할 수 없을 때 머신러닝을 활용하여 클래스를 구분해내기 위해 고안된 신경망입니다. Siamese Network는 두 사진을 입력으로 받아서 벡터화 시킨 후에 두 벡터간의 유사도를 반환합니다. 두 사진이 같을 경우, 유사도를 1로 주고, 다를 경우 유사도를 0으로 주어서 모델을 학습시킵니다.

&nbsp;

**4. Contrastive Loss**\
Contrastive loss는 벡터화된 두 이미지가 같은 클래스일 경우, 두 이미지의 거리를 가깝게 하고, 벡터화된 두 이미자가 다른 클래스일 경우, 두 이미지의 거리를 margin이상이 되도록 합니다.

&nbsp;

**5. VGG(Visual Geometry Group)**\
VGG는 2014 ImageNet Large Scale Visual Recognition Challenge에서 굉장히 좋은 성능을 보여준 CNN망 입니다. 현재 버전으로는 11, 13, 16, 19가 있습니다.

&nbsp;
&nbsp;

### Run :laughing:

1. 데이터 채우기
    * 'data/images_train' directory에 있는 'drowsy'와 'normal'에 데이터를 넣어주셔야 학습이 가능합니다. 
    * 학습된 모델은 'checkpoints' directory에 저장이 됩니다.
    * 'visualization_data/images_train'에 있는 'drowsy'와 'normal'에 데이터를 넣어주셔야 그래프를 그릴 수 있습니다.
    * 녹화된 비디오로 테스트 하고 싶으면 'data/video test' directory에 동영상을 넣어주시면 됩니다.

2. 실행
    * Prompt에서 'python main.py'를 입력하면 프로그램이 실행됩니다.
    * Prompt에서 'embedding_visualization.py'를 입력하면 그래프에 나타난 데이터의 상태를 볼 수 있습니다.
    * 아래에 실행 방법을 간단하게 설명해 놓은 영상입니다.
    
[![trial](https://i.ytimg.com/vi/EBulO0wkQpE/hq1.jpg)](https://youtu.be/EBulO0wkQpE)



&nbsp;
&nbsp;

### Performance :thumbsup:

<img src="https://ifh.cc/g/BaTl7w.png" width="300" height="300">

* 그래프에서 주황색 데이터는 '졸음을 깨기 위한 행동'입니다. 데이터들이 옹기종기 모여있는 모습이 참으로 기특합니다.

* 그래프에서 파란색 데이터는 '일반 운전 행동'입니다. 주황색 보다는 아니지만 그래도 모여있는 모습을 보이고 있습니다. 대견스럽습니다.

* 학습 결과: '졸음을 깨기 위한 행동'과 '일반 운전 행동'이 정도는 다르지만 군집화를 이루었습니다. 그리고 두 군집은 어느 정도 거리를 두고 생성되었습니다. 새로운 데이터가 들어오면 구분이 잘 될 것으로 생각됩니다.

&nbsp;

<img src="https://ifh.cc/g/RBHqHa.png" width="300" height="300">

* Loss는 초반에 급하강 합니다. 그리고 서서히 감소했습니다.

* Loss가 중간에 급격하게 상승하는 경우가 없는 것을 보아 gradient exploding은 없어 보입니다.

* Loss가 중간에 멈추는 경우가 없는 것을 보아 global optimal에 빠지지 않은 것으로 보입니다.


&nbsp;
&nbsp;
&nbsp;

## Test :eyes:

아래 두 개의 동영상으로 모델의 성능을 확인하실 수 있습니다. 클릭!하시면 유튜브로 연결됩니다.

[![test case 1](https://i.ytimg.com/vi/yXfSwO3CmXE/hq1.jpg)](https://youtu.be/yXfSwO3CmXE)\
* 위 동영상은 운전자가 졸음이 오는 것을 느끼고 뒷목을 마사지하는 모습이 담겨있습니다. Dissimilarity가 일반 운전을 할 때, dissimilarity가 1.0 이상으로 출력되고 있습니다. 뒷목을 마사지하는 과정에 dissimilarity가 1.0 이하로 내려갑니다.

&nbsp;

[![test case 2](https://i.ytimg.com/vi/dRNe7GP0RvE/hq1.jpg)](https://youtu.be/dRNe7GP0RvE)\
* 위 동영상은 운전자가 졸음이 오는 것을 느끼고 관자놀이를 누르는 모습이 담겨있습니다. Dissimilarity가 일반 운전을 할 때, dissimilarity가 1.0 이상으로 출력되고 있습니다. 관자놀이를 마사지하는 과정에 dissimilarity가 1.0 이하로 내려갑니다.

&nbsp;
&nbsp;
&nbsp;

## Reference 

### Paper :bookmark_tabs:
* **Eye Blink Detection** - [Eye Blink Detection](https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)
* **Siamese Network** - [Siamese Network](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
* **Contrastive Loss** - [Contrastive Loss](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)

&nbsp;

### Github :computer:
* **Facial Landmarks** - [Facial Landmarks](https://github.com/raja434/driver-fatigue-detection-system)
* **Drowsy Detection** - [Drowsy Detection](https://github.com/AnirudhGP/DrowsyDriverDetection)
* **Siamese Network** - [Siamese Network](https://github.com/tensorfreitas/Siamese-Networks-for-One-Shot-Learning)
* **Contrastive Loss** - [Contrastive Loss](https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch)

&nbsp;
&nbsp;
&nbsp;

## Question_Issues

* 이 프로젝트에 대해서 궁금한 점이 있으신 분은 <conkjh0321@gmail.com>으로 문의해 주시면 답변해 드리겠습니다.

* 동영상에서 뒷목을 마사지하거나 관자놀이를 만질 때 dissimilarity가 1.0 이상 출력되는 경우도 있습니다. 그 이유는 다음과 같습니다. Input 이미지가 2장 필요합니다. A를 카메라에서 계속 들어오는 frame이라고 하면, B는 data set에서 지정한 특정 이미지 한 장 입니다. 따라서, A라는 특정 이미지와 유사한 행동이 B 이미지에 담긴다면 dissimilarity가 작아집니다. 이 문제에 대해서는 10월 말에 업데이트 하겠습니다.

* Data의 양이 적어서 Validation을 진행할 수 없어서 Train과 Test만 진행하였습니다.

&nbsp;
&nbsp;
&nbsp;

## Social

* **My Blog** - [Welcome!](https://conkjh032.tistory.com/) :thumbsup:
* **My Linkedin** - [Welcome double!](https://www.linkedin.com/in/jeonghwan-kim-579242172/) :muscle:

&nbsp;
&nbsp;
