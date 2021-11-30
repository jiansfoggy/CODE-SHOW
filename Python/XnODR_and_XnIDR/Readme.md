[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/xnodr-and-xnidr-two-accurate-and-fast-fully/image-classification-on-mnist)](https://paperswithcode.com/sota/image-classification-on-mnist?p=xnodr-and-xnidr-two-accurate-and-fast-fully)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/xnodr-and-xnidr-two-accurate-and-fast-fully/image-classification-on-cifar-10)](https://paperswithcode.com/sota/image-classification-on-cifar-10?p=xnodr-and-xnidr-two-accurate-and-fast-fully)

# XnODR and XnIDR: Two Accurate and Fast Fully Connected Layers For Convolutional Neural Networks

Although Capsule Networks show great abilities in defining the position relationship between features in deep neural networks for visual recognition tasks, they are computationally expensive and not suitable for running on mobile devices. The bottleneck is in the computational complexity of the Dynamic Routing mechanism used between capsules. On the other hand, neural networks such as XNOR-Net are fast and computationally efficient but have relatively low accuracy because of their information loss in the binarization process. This paper proposes a new class of Fully Connected (FC) Layers by xnorizing the linear projector outside or inside the Dynamic Routing within the CapsFC layer. Specifically, our proposed FC layers have two versions, XnODR (Xnorizing Linear Projector Outside Dynamic Routing) and XnIDR (Xnorizing Linear Projector Inside Dynamic Routing). To test their generalization, we insert them into MobileNet V2 and ResNet-50 separately. Experiments on three datasets, MNIST, CIFAR-10, MultiMNIST validate their effectiveness. Our experimental results demonstrate that both XnODR and XnIDR help networks to have high accuracy with lower FLOPs and fewer parameters (e.g., 95.32\% accuracy with 2.99M parameters and 311.22M FLOPs on CIFAR-10). 

You can find the paper [here](https://arxiv.org/abs/2111.10854).

We upload the code based on different dataset.

To train the model, you can run the following codes directly.

To train on MNIST
```
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/mobilenetv2.py  
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/MobileNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/MobileNet_XnIDR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/ResNet50.py
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/ResNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/ResNet_XnIDR.py
```

To train on CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/MobileNetV2.py
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/MobileNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/MobileNet_XnIDR.py
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/ResNet50.py
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/ResNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/ResNet_XnIDR.py
```

To train on MultiMNIST
```
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/MobileNetV2.py
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/MobileNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/MobileNet_XnIDR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/ResNet50.py
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/ResNet_XnODR.py
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/ResNet_XnIDR.py
```
