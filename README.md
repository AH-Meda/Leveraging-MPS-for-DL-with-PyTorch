# Leveraging MPS for Deep Learning with PyTorch Framework

While NVIDIA CUDA has long been favored for deep learning tasks due to its exceptional performance, Mac users often encounter hardware limitations when running complex models. However, with the introduction of Metal Performance Shaders (MPS) on Apple devices, users can now harness the power of GPU acceleration for their deep learning tasks. All we have to do set up the enviroment, install pytorch to the environment and verfify the installation. 

## Environment Setup

Setting up the environment and running the model is straightforward. Before proceeding with the implementation, ensure you have Xcode installed.

Ensure you have your existing conda environment is ready or create a new one.

    - To activate an existing environment:
        ```
        conda env list
        conda activate <myenv>
        ```
    - Or create a new environment:
        ```
        conda create -n <myenvname> python=3.9
        conda activate <myenvname>
        ```

## Installing PyTorch with MPS Support

To utilize MPS with PyTorch, you need a version of PyTorch compiled with Metal support. Choose the appropriate configuration and execute the provided command. It's recommended to opt for the stable version over the nightly build.

![image](https://github.com/AH-Meda/Leveraging-MPS-for-Deep-Learning-with-PyTorch-/assets/83394560/0cbc777a-52eb-40bd-9b4f-aab4d575985a)


  - Run this output command::
    
        ```
        pip3 install torch torchvision torchaudio
        ```
## Verify the installation
To verify the installation, run the following code snippet in your preferred text editor/IDE within the same environment:
```python
import torch
import math
print(torch.backends.mps.is_available())
```
If the import is successful and the output is True, you are good to go.

## Implementation Example
For demonstration purposes, let's implement a simple Convolutional Neural Network (CNN) using PyTorch for classifying images from the CIFAR-10 dataset.
