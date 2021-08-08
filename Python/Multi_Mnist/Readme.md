# Easy Way to generate multi-mnist dataset

## What is Multi-Mnist ?  
Multi-Mnist dataset is a very important to prove the effectiveness of CapsuleNet. It is generated from 70000 samples in Mnist.  
For each sample of Mnist, which shape is [70000, 28, 28, 1], we combine it with another random sample and generate a new sample with 2 overlapped digits, which is different from placing multiple digits separately into one image. Its shape is [36, 36, 1]. Then we repeat this process 1000 times. Therefore, Multi-Mnist totally has 70M samples.

## How to get this idea?  
However, this dataset is not ready to download directly like Mnist or Cifar-10, researchers have to generate it firstly before experiment. The current methods published on github is not friendly to beginners. It takes time for readers to understand it. To reduce the burden on reading code, we propose an easy way to generate Multi-Mnist dataset. Our method is a sort of plug-in function, and it suits for both 2 popular framework, Tensorflow and PyTorch. Users can implement it by directly copy and paste it into the their code.

## Difference between 2 uploaded files.  
### slow_generate_multimnist.py 
We design a very specific structure there. 

* Fuse images from train and test set to generate a new image set, so does label set.
* Sort image set based on class order and split the image set into 10 subsets. Each subset represents 1 category. So does 10 label subsets.
* Start from class \textit{i} and select related subset and count its sample number as num_0, then pick one out of 10 subsets, 



fast_generate_multimnist.py

slow_generate_multimnist.py
