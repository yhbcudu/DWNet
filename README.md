Dual-Branch D-WNet: A Classification Method for Cloudy Remote Sensing Images Integrating Optical and Time-Series SAR Features

<img width="865" height="634" alt="image" src="https://github.com/user-attachments/assets/a2e84c48-8202-475d-9e92-af4b11782c9b" />

This repository contains the official PyTorch implementation of D-WNet, as described in the paper: "Dual-Branch D-WNet: A Classification Method for Cloudy Remote Sensing Images Integrating Optical and Time-Series SAR Features".

1 Introduction

D-WNet is a hierarchical cross-modal fusion framework designed for robust Land Use and Land Cover (LULC) classification in persistently cloudy and rainy regions.

Traditional optical remote sensing is often hindered by cloud cover, while SAR data suffers from speckle noise and lower resolution. D-WNet addresses these challenges by explicitly decoupling and adaptively synergizing heterogeneous multimodal inputs:

1.  Optical Branch: Utilizes Depthwise Separable Convolutions (DSC) to extract high-resolution spatial-spectral features efficiently.
2.  SAR Branch: Employs ConvLSTM and a Temporal Attention Module to model the temporal evolution of SAR backscatter (Sentinel-1) over a 12-month cycle.
3.  Enhanced Adaptive Feature Fusion (Enhanced AFF): A novel module integrating Dense Feature Aggregation, Local-Global Dual-Attention, and Subspace Decoupling to dynamically balance modality contributions based on cloud conditions.

Experiments on Zaling-Eling Lake, Gansu, and California datasets demonstrate that D-WNet achieves OA improvements of 6.8–12.4% over state-of-the-art methods.

2 Architecture

(Note: Replace with Figure 2 from the manuscript)

The network consists of three key components:

Dual-Branch Encoder: Decouples static optical textures from dynamic SAR scattering.
Enhanced AFF Module: Dynamically reweights features to rely on SAR when optical data is degraded by clouds.
<img width="864" height="553" alt="image" src="https://github.com/user-attachments/assets/76311d50-3e7d-4f18-bc4c-80353d0202b5" />

Residual-Guided Decoder: Preserves boundary details and suppresses noise.

3 Data Availability

The data utilized in this study are openly available. Specifically, the Sentinel-1 SAR data and Sentinel-2 optical imagery can be accessed via the Copernicus Data Space Ecosystem. The NLCD and GLC-FCS30 datasets can be accessed and downloaded from the following links:

Sentinel-1 SAR data: [Copernicus Data Space](https://dataspace.copernicus.eu/data-collections/sentinel-data/sentinel-1)
Sentinel-2 optical imagery: [Copernicus Data Space](https://dataspace.copernicus.eu/data-collections/copernicus-sentinel-data/sentinel-2)
NLCD (National Land Cover Database): [MRLC.gov](https://www.mrlc.gov/data) 
GLC-FCS30 (Global Land Cover with Fine Classification System): [Casearth Data Sharing Platform](https://data.casearth.cn/thematic/glc_fcs30) 
(Note: Preprocessing steps such as terrain correction and Lee filtering were performed using Google Earth Engine as described in the paper.)


4  Usage

 1. Data Preparation

Organize your dataset into the following structure:

dataset/
├── train/
│   ├── optical/       (B, 4, H, W)
│   ├── sar_seq/       (B, T, 2, H, W)
│   └── label/         (B, H, W)
└── test/
    ├── optical/
    ├── sar_seq/
    └── label/
```

 2. Training

To train the D-WNet model, run:

```bash
python train.py --batch_size 16 --epochs 100 --lr 3e-4 --dataset ZalingLake
```

Note: The training process follows a three-stage strategy: (1) Optical pre-training, (2) Joint training, and (3) Fusion fine-tuning.

 3. Inference / Testing

To evaluate the model on the test set:

```bash
python test.py --weights checkpoints/best_model.pth
```

Comparison Methods

This repository also includes implementations of the following baseline methods used for comparison in the paper :

   Single-Source: U-Net, U-Net3+, L-UNet, ConvLSTM.
   Multi-Source:
       ConcatenationNet (Gross et al., 2020)
       FeatureWeightNet (Yang et al., 2023)
       WeightedVoteNet (Kumar et al., 2024)
       CrossAttentionNet (Zeng et al., 2024)
       KCCA (Ren et al., 2024)

5 Citation

If you find this work or code useful for your research, please cite our paper:

```bibtex
@article{yao2025dwnet,
  title={Dual-Branch D-WNet: A Classification Method for Cloudy Remote Sensing Images Integrating Optical and Time-Series SAR Features},
  author={Yao, Haobo and Dou, Peng and Zhang, Lifeng and Huang, Chunlin and He, Yi and Zhang, Ying and Hou, Jinliang and Zhang, Mingwang and Guo, Jifu and Ye, Jiaqi and Li, Liang},
  journal={Journal Name (To be updated)},
  year={2025}
}
```
