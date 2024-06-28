# Glance-Focus
<p align="left">
    <a href='https://arxiv.org/abs/2401.01529'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://openreview.net/pdf?id=J6Niv3yrMq'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=plastic&logo=adobeacrobatreader&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://www.python.org/'>
      <img src='https://img.shields.io/badge/python-3.7-blue.svg' alt='Python'>
    </a>
</p>

This repo contains source code for our NeurIPS 2023 paper:

[Glance and Focus: Memory Prompting for Multi-Event Video Question Answering](https://openreview.net/forum?id=J6Niv3yrMq)

[Ziyi Bai](https://scholar.google.com/citations?hl=zh-CN&user=jRe11usAAAAJ), [Ruiping Wang](https://scholar.google.com/citations?hl=zh-CN&user=duIUwpwAAAAJ), [Xilin Chen](https://scholar.google.com/citations?user=vVx2v20AAAAJ)

![overview](https://github.com/ByZ0e/Glance-Focus/blob/main/overview.png)

## Updates 🎉

✅ AGQA v2 Benchmark Supported

✅ EgoTaskQA Benchmark Supported

✅ NExT-QA Benchmark Supported

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

## Data Preparation
You need to obtain necessary dataset and features. You can choose one of the following options to do so:

#### Option 1: Download Features from Our Shared Drive
You can download the dataset annotation files and features directly to the `DEFAULT_DATASET_DIR`.\
We currently upload all necessary files for running on STAR benchmark. You can download from [Google Drive](https://drive.google.com/file/d/11sI_iW_42yetN2U8WdwsdARmQPhdhQht/view?usp=sharing).

It should have the following structure:
```
├── /STAR/
│  ├── /txt_db/
│  │  ├── action_mapping.txt
│  │  ├── events.json
│  │  ├── test.jsonl
│  │  ├── train.jsonl
│  │  └── val.jsonl
│  ├── /vis_db/
│  │  ├── s3d.pth
│  │  └── strID2numID.json
```

We currently upload all necessary files for running on AGQA benchmark. You can download from [Google Drive](https://drive.google.com/file/d/1VQ1mE9bM3FkCNmBe5JafkoZYI0AaweNA/view?usp=sharing).

It should have the following structure:
```
├── /AGQA/
│  ├── /txt_db/
│  │  ├── action_mapping.txt
│  │  ├── charades.json
│  │  ├── test_balanced.txt
│  │  ├── train_balanced.txt
│  │  └── agqa_balanced_txt_vocab.json
│  ├── /vis_db/
│  │  └── frcnn.pth
```

We currently upload all necessary text files for running on EgoTaskQA benchmark. You can download from [Google Drive](https://drive.google.com/file/d/1n2u8Cd-yCJeltwuzMJkW-cSgPpiXzERr/view?usp=sharing).

It should have the following structure:
```
├── /EgoTaskQA/
│  ├── /txt_db/
│  │  ├── hois_full_trans.json
│  │  ├── events.json
│  │  ├── formatted_train_qas_encode.json
│  │  ├── formatted_val_qas_encode.json
│  │  ├── formatted_test_qas_encode.json
│  │  └── answer_set.txt
```

We currently upload all necessary text files for running on NExT-QA benchmark. You can download from [Google Drive](https://drive.google.com/file/d/1g5i6Di50WFsl5vgiNUlffcjqLRUX9TWs/view?usp=sharing).

It should have the following structure:
```
├── /NExT-QA/
│  ├── /txt_db/
│  │  ├── train_mc.csv
│  │  ├── val_mc.csv
│  │  ├── test_mc.csv
│  │  └── vocab.json
```

#### Option 2: Extract Features Using Provided Script

If you wish to reproduce the data preprocessing and video feature extraction procedures.

1. Download Raw Data
- **STAR**: Download it from [the data providers](https://github.com/csbobby/STAR_Benchmark). 
- **AGQA**: Download it from [the data providers](https://github.com/madeleinegrunde/AGQA_baselines_code).
- **EgoTaskQA**: Download it from [the data providers](https://sites.google.com/view/egotaskqa). 
- **NExT-QA**: Download it from [the data providers](https://github.com/doc-doc/NExT-QA).

2. Data Preprocessing
- Please follow the data format in **Option 1** to preper the corresponding data.
- We also plan to upload the corresponding data processing code for each benchmark.

3. Extract video features
We follow the recent works to extract the video features. Here are some reference code:
- **S3D feature**: Please refer to [Just-Ask](https://github.com/antoyang/just-ask).
- **C3D feature**: Most of the benchmarks have provided this feature, please refer to the original benchmarks.
- **CLIP feature**: Please refer to [MIST](https://github.com/showlab/mist).

## Training
With your environment set up and data ready, you can start training the model.

We support both **unsupervised** and **supervised** setting training, since some VideoQA benchmarks like NExT-QA do not provide event-level annotations.

- unsupervised setting
```
python train_glance_focus_uns.py --basedir expm/star --name gf_logs --device_id 0 --test_only 0 \
--qa_dataset star --base_data_dir $DEFAULT_DATASET_DIR \
--losses_type ['qa','cls','giou','cert']
```
- supervised setting
```
python train_glance_focus_sup.py --basedir expm/star --name gf_logs --device_id 0 --test_only 0 \
--qa_dataset star --base_data_dir $DEFAULT_DATASET_DIR \
--losses_type ['qa','cls','l1']
```

For AGQA benchmark, some parameters are different.

Note that AGQA do not have validation set, you may split it from training set by yourself.

What's more, remember to change to the AGQA dataloader.
```
python train_glance_focus_sup.py --basedir expm/agqa --name gf_logs --device_id 0 --test_only 0 \
--qa_dataset agqa --base_data_dir $DEFAULT_DATASET_DIR \
--batch_size 128 --nepoch 20 --lr 5e-5 --i_test 9000 --i_weight 9000 --output_dim 157 \
--train_data_file_path '{}/txt_db/train_balanced.txt'
--test_data_file_path '{}/txt_db/test_balanced.txt'
--event_anno_path '{}/txt_db/charades.json'
--app_feat_path '{}/vis_db/frcnn.pth'
--feature_dim 512
--ans2label_path '{}/txt_db/agqa_balanced_txt_vocab.json'
--losses_type ['qa','cls','l1']
```

For EgoTaskQA and NExT-QA benchmarks, their data formats are slightly different from STAR and AGQA. 

We release the corresponding data loaders we used.

You may have to edit the data loading code.

## Available checkpoints
Supervised trained on STAR dataset. Download from [Google Drive](https://drive.google.com/file/d/1oZHqHQI9rUCpKIwJQvVQf4sNyeu1E_Du/view?usp=sharing).

## Inference
```
python train_glance_focus_uns.py --device_id 0 --test_only 1 \
--qa_dataset star --base_data_dir $DEFAULT_DATASET_DIR \
--reload_model_path expm/star/gf_logs/ckpts_2024-01-17T10-30-46/model_3000.tar \
```

## Ackonwledgements
We are grateful to [Just-Ask](https://github.com/antoyang/just-ask), [MIST](https://github.com/showlab/mist) and [ClipBERT](https://github.com/jayleicn/ClipBERT), on which our codes are developed.

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



