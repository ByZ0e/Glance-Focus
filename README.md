# Glance-Focus
<p align="left">
    <a href='https://openreview.net/pdf?id=J6Niv3yrMq'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
</p>

This repo contains source code for our NeurIPS 2023 paper:

[Glance and Focus: Memory Prompting for Multi-Event Video Question Answering](https://openreview.net/forum?id=J6Niv3yrMq)

[Ziyi Bai](https://scholar.google.com/citations?hl=zh-CN&user=jRe11usAAAAJ), [Ruiping Wang](https://scholar.google.com/citations?hl=zh-CN&user=duIUwpwAAAAJ), [Xilin Chen](https://scholar.google.com/citations?user=vVx2v20AAAAJ)

![overview](https://github.com/ByZ0e/Glance-Focus/blob/main/overview.png)

## Prerequisites

The project requires the following:

1. **PyTorch** (version 1.9.0 or higher): The project was tested on PyTorch 1.11.0 with CUDA 11.3 support.
2. **Hardware**: We have performed experiments on NVIDIA GeForce RTX 3090Ti with 24GB GPU memory. Similar or higher specifications are recommended for optimal performance.
3. **Python packages**: Additional Python packages specified in the `requirements.txt` file are necessary. Instructions for installing these are given below.

## Setup Instructions
Let's begin from creating and activating a Conda environment an virtual environment 
```
conda create --name gfenv python=3.7
conda activate gfenv
```
Then, clone this repository and install the requirements.
```
$ git clone https://github.com/ByZ0e/Glance-Focus.git
$ cd Glance-Focus
$ pip install -r requirements.txt
```

## Training
With your environment set up and data ready, you can start training the model. To begin training, run `train_glance_focus_uns.py` for unsupervised setting or run `train_glance_focus_sup.py` for supervised setting.

## Citation
If you find our paper and/or code helpful, please consider citing:
```
@inproceedings{bai2023glance,
  title={Glance and Focus: Memory Prompting for Multi-Event Video Question Answering},
  author={Bai, Ziyi and Wang, Ruiping and Xilin, CHEN},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```



