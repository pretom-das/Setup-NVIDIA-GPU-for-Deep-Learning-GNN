# Setup-NVIDIA-GPU-for-Deep-Learning-GNN

## Step 1: Cheack GPU support

```bash
nvidia-smi

```

## Step 2: NVIDIA Video Driver

You should install the latest version of your GPUs driver. You can download drivers here:
 - [NVIDIA GPU Drive Download](https://www.nvidia.com/Download/index.aspx)

## Step 3: Visual Studio C++ and vscode

You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options.
 - [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

You will need Visual Studio code, 
 - [Visual Studio Code](https://code.visualstudio.com/download)

## Step 4: Anaconda/Miniconda

You will need anaconda to install all deep learning packages
 - [Download Anaconda](https://www.anaconda.com/download/success)
Miniconda
 - [Download miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main)

## Step 4: CUDA Toolkit

 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

## Step 5: cuDNN

 - [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)


## Step 6: Install PyTorch 

 - [Install PyTorch](https://pytorch.org/get-started/locally

 ## Step 7: Activate virtual env

```bash
conda config --add channels conda-forge
conda config --set channel_priority flexible

```


```bash
conda create -n gnn_env python=3.12
conda activate gnn_env

```


## Finally run the following script to test your GPU

```python
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
