# Deep Convolution Network for Direction of Arrival Estimation With Sparse Prior

## 基本信息

期刊：IEEE Signal Processing Letters

年份：2019

中科院分区/影响因子：二区 / 3.201

## 摘要

主要内容：提出了一种深度学习框架的DOA估计方法。首先说明阵列麦克风输入信号的协方差矩阵的列可以表示为空间频谱的欠采样噪声线性测量。然后提出了一种从大量训练数据中学习的深度卷积网络（DCN）。

优势：1. 与传统的稀疏诱导方法相比，可以获得近乎实时的DOA估计；2.与现有的基于深度学习的方法相比，使用稀疏先验提升了DOA估计的性能； 3.仿真结果证明了所提方法在DOA估计精度和计算效率方面的优越性，尤其是在低信噪比的情况下。

## 引言

### 研究现状

1. DL techniques：深度学习技术
   1. 处理单信号场景或在网格间距非常大的情况下定位声源
   2. 几乎不能用于电磁（EM）DOA估计
2. 刘等人提出了一种端到端（End-to-end）神经网络，通过一系列分类器在预定义的网络上检测（EM）信号的存在
3. 陈等人通过自编码器在VHF雷达中提取不同方向的特征，而不是直接估计DOA
4. 黄等人提出了一种用于超分辨率DOA估计的DNN方案

### 方法原理

DOA估计可以看作是稀疏线性逆问题在压缩感知中的应用，因此提出一个带有卷积层的深度网络来学习从输入数据到DOA谱的逆变换

### 创新点

1. 考虑了声源信号的空间稀疏性
2. 与全连接神经网络相比，在整个感受野共享权重的稀疏卷积具有更快的学习速度

## 问题建模

![1669558692264](image/readme/1669558692264.png)

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/fc05fff4-9b91-4f51-b613-1a82eee54099.png)

## 实验验证

### 数据集创建

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/4913d203-6bc2-43c9-85be-deec19ea9f74.png)

### 实验算法

1. CNN + ReLU（Proposed）
2. CNN + Tanh
3. DNN + （ReLU / Tanh）
4. Autoencoder （参考文献[5]）
5. SBL / SBLR （稀疏贝叶斯学习模型）

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/9421a5a5-9ea8-4ec6-b09c-21e4c7c006af.png)

### 实验结果

1. 训练验证 MSE Loss 曲线

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/bc622de0-c8c8-46f0-9bf5-63adacc61f61.png)

结论：可以看出在训练和验证时，说提出方法的MSE LOSS曲线均最低

2. 重构的信号谱

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/e5876558-f9d1-49d7-8862-e695b4061d27.png)

结论：所提出方法在声源角度很接近的时候，仍然有很好的性能（对比SBL，其在声源接近时有明显的旁瓣）。而且其重构的信号谱没有毛刺（对比Autoencoder）。

3. 训练耗时

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/5aabd199-92ad-4743-95d0-fcafa7342f1b.png)

结论：所提方法训练时间和测试时间都很低，可以实现近乎实时的DOA估计

4. DOA估计以及估计误差

结论：所提方法其DOA估计与真实值非常吻合，绝大多数误差都小于2.5度。SBL和Autoencoder方法则有时无法恢复声源信号，尤其是在声源接近时不够稳健

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/dc362d47-b87e-4853-9b5a-a593cfbef86f.png)

5. 声源数量对DOA估计的影响

结论：所有算法均能实现良好的DOA估计， 但使用Autoencoder时毛刺较多

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/74a4d285-1bb2-4399-96f1-a05315a62d05.png)

6. 信噪比（SNR）以及信号间角度对DOA估计的影响

![](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/7jP2lRj7REG6l8g5/img/501908ec-9b6f-44ab-889e-64fff1cef3f6.png)

结论：当信噪比低以及声源间角小时，所提方法在估计精度中有着显著的优势；当信噪比大以及声源间角大时，所提方法与SBL方法有着相近的性能，但其计算消耗要小得多。

## 结论

本文提出了一种基于 DCN 的高效空间谱恢复算法，并将其应用于 EM DOA 估计。首先通过引入空间过完备公式将 DOA 估计问题转换为稀疏线性逆问题。然后介绍了DCN的结构设计和训练过程。与传统的基于迭代的稀疏恢复算法相比，基于DCN的方法只需要前馈计算，从而可以实现实时DOA估计。此外，卷积层的学习和泛化能力以及 ReLU 激活有助于它在 SNR 较低或信号间角度分离较小时实现具有竞争力甚至更好的 DOA 估计性能。仿真和实验清楚地验证了所提方法的优越性
