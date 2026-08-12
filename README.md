# Secure and Explainable Deep Learning Models and Transformers for Healthcare Diagnostics: A Case Study on Diabetic Retinopathy

Code accompanying the MRes Artificial Intelligence dissertation.

**Author:** Tochukwu Edith Okafor (Student ID: 2049720)
**Institution:** School of Architecture, Computing and Engineering, University of Wolverhampton
**Supervisor:** Hiran Patel

---

## Overview

This repository contains the code for a cross-architecture, cross-method study of explainability in diabetic retinopathy (DR) models. The central question is whether post-hoc explainability methods behave consistently across architecturally distinct deep learning models applied to the same task. The work runs along two strands that share the same architectural logic (a network trained from scratch, pretrained convolutional networks, and vision transformers) and the same circular-masking experiment:

1. **Classification and explainability** on a 5-class Kaggle DR dataset.
2. **Lesion segmentation** on the IDRiD dataset.

The main findings: gradient-based explanation methods are faithful on convolutional networks but degrade on transformers, LIME is the most architecture-agnostic of the four methods tested, and circular masking relocates boundary-focused attention rather than removing it.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── 00_leakage_audit/01_build_clean_split       # Perceptual-hash duplicate audit and deduplicated group-aware split
├── 02a_retrain_cnns/02b_retrain_transformers/03a_cnns_seeded/03b_swin_seeded/03b2_deit_seeded/04a_cnns_class_imbalance_handling/04b_swin_class_imbalance_handling/04b2_deit_class_imbalance_handling       # Five architectures, training, multi-seed protocol, imbalance ablation
├── 06_build_xai_image_set/07a_xai_vanilla_cnn/07b_xai_inception/07c_xai_convnext/07d_xai_deit/07e_xai_swin/08_xai_iou/08b_xai_iou_significance_wilcoxon_agreement    # Grad-CAM, saliency, SHAP, LIME, attention rollout/windowed attention
│                           # plus deletion/insertion faithfulness and IoU agreement metrics
├── 09_build_masked_dataset/10a_masked_vanilla_cnn/10b_masked_inception/10c_masked_convnext/10d_masked_swin/10e_masked_deit/11_masked_vs_unmasked_metrics/13_masking_gradcam_figure             # Circular retinal masking experiment (radius-scaling, retraining)
├── 15_idrid_preprocessing/16a_seg_unet_scratch_unmasked/16a_seg_unet_scratch_masked/16b_seg_unet_pretrained_unmasked/16b_seg_unet_pretrained_masked/16c_seg_segformer_unmasked/16c_seg_segformer_masked/17_segmentation_metrics       # IDRiD tiling, three U-Net encoders, Dice + weighted BCE, per-lesion metrics
└── 08_xai_iou/08b_xai_iou_significance_wilcoxon_agreement             # Exploratory and analysis notebooks (incl. IoU / Wilcoxon agreement analysis)
```



---

## Datasets

The datasets are **not** included in this repository. They are publicly available and should be downloaded from their original sources:

- **Kaggle DR dataset** (5-class grading, 3,554 fundus images): `shajinrp/diabetic-retinopathy` on Kaggle.
- **IDRiD** (Indian Diabetic Retinopathy Image Dataset, pixel-level lesion masks): Porwal et al. (2018), available from the IEEE DataPort / IDRiD grand-challenge page.

All experiments were run in Kaggle's GPU compute environment.

---

## Environment and installation

Developed in Python 3 with PyTorch. To reproduce the environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Pinned package versions are listed in `requirements.txt`. A CUDA-capable GPU is recommended; the reported experiments used Kaggle GPU runtimes.

---

## Reproducing the study

The pipeline follows the order of the numbered folders.

1. **Leakage audit and split** (`00_leakage_audit/01_build_clean_split`)
   Computes a perceptual hash per image, detects near-duplicate clusters with a Hamming-distance threshold of 2, groups them with union-find, and writes a deduplicated, group-aware, stratified 70/15/15 split. The saved split indices are reused unchanged by every downstream model.

2. **Classification** (`02a_retrain_cnns/02b_retrain_transformers/03a_cnns_seeded/03b_swin_seeded/03b2_deit_seeded/04a_cnns_class_imbalance_handling/04b_swin_class_imbalance_handling/04b2_deit_class_imbalance_handling`)
   Trains the five retained architectures (custom CNN from scratch, Inception V3, ConvNeXt-Tiny, DeiT-Base, Swin-Base) on the deduplicated split. Convolutional models use five seeds, transformers three. Augmentation is applied to every run; class-imbalance handling is the single varied condition (weighted-loss ablation). Each run saves weights and test-set probabilities.

3. **Explainability** (`06_build_xai_image_set/07a_xai_vanilla_cnn/07b_xai_inception/07c_xai_convnext/07d_xai_deit/07e_xai_swin/08_xai_iou/08b_xai_iou_significance_agreement`)
   Applies Grad-CAM, saliency, SHAP and LIME (plus attention rollout for DeiT and windowed attention for Swin) to a fixed 30-image set, six per grade, using each model's median-accuracy seed. Computes deletion/insertion faithfulness curves and top-k IoU agreement, including the paired Wilcoxon signed-rank test on LIME-vs-gradient agreement.

4. **Masking** (`09_build_masked_dataset/10a_masked_vanilla_cnn/10b_masked_inception/10c_masked_convnext/10d_masked_swin/10e_masked_deit/11_masked_vs_unmasked_metrics/13_masking_gradcam_figure`)
   Locates the retinal disc, scales its radius by 0.9 to remove the outer boundary ring, retrains each architecture on the masked images with the matched seed, and regenerates the explanations for a before/after comparison.

5. **Segmentation** (`15_idrid_preprocessing/16a_seg_unet_scratch_unmasked/16a_seg_unet_scratch_masked/16b_seg_unet_pretrained_unmasked/16b_seg_unet_pretrained_masked/16c_seg_segformer_unmasked/16c_seg_segformer_masked/17_segmentation_metrics`)
   Tiles IDRiD images into 512x512 patches at native resolution, trains three U-Net models (ResNet-34 from scratch, ResNet-34 pretrained, MiT-B2 / SegFormer) with a Dice + weighted-BCE loss, and evaluates Dice, IoU, sensitivity and precision per lesion type. Repeats under masking.

---

## Key libraries

PyTorch, timm (vision transformer and CNN backbones), segmentation-models-pytorch (U-Net encoders), SHAP, LIME, OpenCV and scikit-image (masking and perceptual hashing), scikit-learn and SciPy (metrics and the Wilcoxon test), NumPy, pandas and Matplotlib.

---

## Notes on reproducibility

Each run seeds the Python, NumPy and PyTorch random number generators and the data-loader generator, and places cuDNN in deterministic mode. Run-to-run variation on this dataset is nonetheless substantial for the convolutional models, which is why results are reported across multiple seeds rather than from single runs.

---

## Citation

If you refer to this work, please cite the dissertation:

Okafor, T.E. (2026) *Secure and Explainable Deep Learning Models and Transformers for Healthcare Diagnostics: A Case Study on Diabetic Retinopathy.* MRes dissertation, University of Wolverhampton.

---

## License

This code is released for academic and review purposes. The datasets referenced above are subject to their own licenses and terms of use.
