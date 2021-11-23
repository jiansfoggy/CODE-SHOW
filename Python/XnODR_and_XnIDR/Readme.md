# XnODR and XnIDR: Two Accurate and Fast Fully Connected Layers For Convolutional Neural Networks

Although Capsule Networks show great abilities in defining the position relationship between features in deep neural networks for visual recognition tasks, they are computationally expensive and not suitable for running on mobile devices. The bottleneck is in the computational complexity of the Dynamic Routing mechanism used between capsules. On the other hand, neural networks such as XNOR-Net are fast and computationally efficient but have relatively low accuracy because of their information loss in the binarization process. This paper proposes a new class of Fully Connected (FC) Layers by xnorizing the linear projector outside or inside the Dynamic Routing within the CapsFC layer. Specifically, our proposed FC layers have two versions, XnODR (Xnorizing Linear Projector Outside Dynamic Routing) and XnIDR (Xnorizing Linear Projector Inside Dynamic Routing). To test their generalization, we insert them into MobileNet V2 and ResNet-50 separately. Experiments on three datasets, MNIST, CIFAR-10, MultiMNIST validate their effectiveness. Our experimental results demonstrate that both XnODR and XnIDR help networks to have high accuracy with lower FLOPs and fewer parameters (e.g., 95.32\% accuracy with 2.99M parameters and 311.22M FLOPs on CIFAR-10). 

You can find the paper [here](https://arxiv.org/abs/2111.10854).

We upload the code based on different dataset.

To train the model, you can run the following codes directly.

CUDA_VISIBLE_DEVICES=0 python3 ./MNIST/filename.py  
CUDA_VISIBLE_DEVICES=0 python3 ./CIFAR-10/filename.py  
CUDA_VISIBLE_DEVICES=0 python3 ./MultiMNIST/filename.py
