# CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection

[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](https://papers.miccai.org/miccai-2026/paper/3797_paper.pdf)
[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--3--032--38072--2__9-blue)](https://doi.org/10.1007/978-3-032-38072-2_9)
[![arXiv](https://img.shields.io/badge/arXiv-2606.04898-b31b1b)](https://arxiv.org/abs/2606.04898)
[![Project page](https://img.shields.io/badge/Project-Page-green)](https://malga-vision.github.io/CDPM-Align_AcademicPage/)

**Roberto Di Via, Irina Voiculescu, Francesca Odone, Vito Paolo Pastore**
MaLGa, DIBRIS, University of Genoa, Italy · Department of Computer Science, University of Oxford, UK

Official code for the MICCAI 2026 paper **"CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection"**.

[Paper (PDF)](https://papers.miccai.org/miccai-2026/paper/3797_paper.pdf) · [Springer (LNCS 16881, pp. 89–99)](https://doi.org/10.1007/978-3-032-38072-2_9) · [arXiv:2606.04898](https://arxiv.org/abs/2606.04898) · [Project page](https://malga-vision.github.io/CDPM-Align_AcademicPage/)

See also our WACV 2027 follow-up [PACER](https://github.com/Malga-Vision/PACER-wacv27), a curriculum self-training stage that can finetune a CDPM-Align detector on an unlabeled pool.

---

## Overview

We compare pretraining strategies for few-shot anatomical landmark detection across three public benchmarks:

| Dataset | Landmarks | Metric units | Default shots |
|---|---|---|---|
| Shenzhen (chest) | 6 | pixels | 10, 25 |
| ISBI2015 (cephalo) | 19 | mm | 10, 25 |
| DHA (hand) | 37 | mm | 10, 25 |

| Method | Paper name | Config prefix |
|---|---|---|
| CDPM-Align (ours) | CDPM-align (λ=5) | `cdpm_align` |
| CDPM — NIH pretraining | CDPM (NIH, λ=0) | `cdpm_nih` |
| DINO SSL | ResNet-101 (DINO) | `ssl_dino` |
| MoCo v3 SSL | ResNet-101 (MoCo v3) | `ssl_mocov3` |
| SimCLR v2 SSL | ResNet-101 (SimCLR v2) | `ssl_simclrv2` |
| ImageNet init | ResNet-101 (ImageNet) | `imagenet` |
| DDPM DiVia | Di Via et al. | `ddpm_divia` |
| CDPM, random init | CDPM Scratch | `cdpm_scratch` |

The Zhu et al. [32] baseline (universal multi-anatomy detector, "You Only Learn Once") is not in this repository. It was run with the authors' official code: [MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection).

---

## Pretraining

All configs use relative paths. Datasets are read from `data/` (see [Datasets](#datasets)) and outputs are written under `outputs/`. Run every command from the repository root, or edit the paths in the configs.

### Phase 1 — CDPM base (combined chest + hand + cephalo, 45k iterations, batch size 16)

```bash
python -m pretraining.cdpm.run -c configs/pretraining/cdpm_combined.json
```

### Phase 1 — CDPM base (NIH ChestX-ray14, 50k iterations)

```bash
python -m pretraining.cdpm.run -c configs/pretraining/cdpm_nih.json
```

### Phase 2 — CDPM-Align fine-tuning (5k iterations, λ=5, batch size 16, lr 1e-5)

Phase 1 plus Phase 2 gives the paper's schedule: 50k iterations at batch size 16 (45k diffusion + 5k alignment). The paper does not give the learning rate of the alignment phase; the config uses 1e-5.

```bash
python -m pretraining.cdpm_align.run_finetune \
    -c configs/pretraining/cdpm_align.json \
    --lambda_align 5.0 \
    --pretrained_checkpoint checkpoints/pretraining/cdpm/combined_45k.pt
```

### SSL — DINO

```bash
python -m pretraining.ssl.ssl_main -c configs/pretraining/ssl_dino.json
```

### SSL — MoCo v3

```bash
python -m pretraining.ssl.ssl_main -c configs/pretraining/ssl_mocov3.json
```

### SSL — SimCLR v2

```bash
python -m pretraining.ssl.ssl_main -c configs/pretraining/ssl_simclrv2.json
```

### DDPM DiVia (domain-specific, one model per dataset)

```bash
# Chest (Shenzhen)
python -m pretraining.ddpm_divia.ddpm_main -c configs/pretraining/ddpm_config.json

# Hand (DHA)
python -m pretraining.ddpm_divia.ddpm_main -c configs/pretraining/ddpm_config.json \
    --dataset hand

# Cephalo (ISBI2015)
python -m pretraining.ddpm_divia.ddpm_main -c configs/pretraining/ddpm_config.json \
    --dataset cephalo
```

---

## Landmark Detection

All downstream experiments use the NLL loss, AdamW (lr=1e-4), an effective batch size of 8, up to 200 epochs, and early stopping (patience=15).
Default: **10 training samples**. Override with `-n N` (e.g., `-n 25`).

Each run writes to `<output_dir>/<dataset>/<model>/<weights>/<batch_lr>/<loss>/samples_<N>_total/<config name>/seed<seed>/`. Because the config name and seed are in the path, different configs (for example `cdpm_align` and `cdpm_nih`) and different seeds never share a directory. If `models/best_checkpoint.pt` already exists in that directory, training is skipped and the existing model is only evaluated. Test metrics go to `landmarks_results.txt` in the same directory.

**5-run protocol.** The paper reports mean ± std over 5 independent runs. The seeds used for the paper are not recorded in this repository. Repeat each command with 5 different seeds via `-s`, for example:

```bash
for seed in 0 1 2 3 4; do
    python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_align.json -n 10 -s $seed
done
```

The seed controls initialisation, augmentation and data order. The few-shot training images are the same for every seed (see [Datasets](#datasets)).

### Shenzhen (chest)

```bash
# CDPM-Align (ours) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_align.json

# CDPM-Align — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_align.json -n 25

# CDPM (NIH, λ=0) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_nih.json

# CDPM (NIH, λ=0) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_nih.json -n 25

# Di Via et al. (DDPM) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ddpm_divia.json

# Di Via et al. (DDPM) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ddpm_divia.json -n 25

# ResNet-101 ImageNet — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/imagenet.json

# ResNet-101 ImageNet — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/imagenet.json -n 25

# ResNet-101 DINO — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_dino.json

# ResNet-101 DINO — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_dino.json -n 25

# ResNet-101 MoCo v3 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_mocov3.json

# ResNet-101 MoCo v3 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_mocov3.json -n 25

# ResNet-101 SimCLR v2 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_simclrv2.json

# ResNet-101 SimCLR v2 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/ssl_simclrv2.json -n 25

# CDPM Scratch (random init) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_scratch.json

# CDPM Scratch (random init) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/chest/cdpm_scratch.json -n 25
```

### ISBI2015 (cephalo)

```bash
# CDPM-Align (ours) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_align.json

# CDPM-Align — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_align.json -n 25

# CDPM (NIH, λ=0) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_nih.json

# CDPM (NIH, λ=0) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_nih.json -n 25

# Di Via et al. (DDPM) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ddpm_divia.json

# Di Via et al. (DDPM) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ddpm_divia.json -n 25

# ResNet-101 ImageNet — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/imagenet.json

# ResNet-101 ImageNet — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/imagenet.json -n 25

# ResNet-101 DINO — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_dino.json

# ResNet-101 DINO — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_dino.json -n 25

# ResNet-101 MoCo v3 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_mocov3.json

# ResNet-101 MoCo v3 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_mocov3.json -n 25

# ResNet-101 SimCLR v2 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_simclrv2.json

# ResNet-101 SimCLR v2 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/ssl_simclrv2.json -n 25

# CDPM Scratch (random init) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_scratch.json

# CDPM Scratch (random init) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/cephalo/cdpm_scratch.json -n 25
```

### DHA (hand)

```bash
# CDPM-Align (ours) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_align.json

# CDPM-Align — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_align.json -n 25

# CDPM (NIH, λ=0) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_nih.json

# CDPM (NIH, λ=0) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_nih.json -n 25

# Di Via et al. (DDPM) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ddpm_divia.json

# Di Via et al. (DDPM) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ddpm_divia.json -n 25

# ResNet-101 ImageNet — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/imagenet.json

# ResNet-101 ImageNet — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/imagenet.json -n 25

# ResNet-101 DINO — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_dino.json

# ResNet-101 DINO — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_dino.json -n 25

# ResNet-101 MoCo v3 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_mocov3.json

# ResNet-101 MoCo v3 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_mocov3.json -n 25

# ResNet-101 SimCLR v2 — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_simclrv2.json

# ResNet-101 SimCLR v2 — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/ssl_simclrv2.json -n 25

# CDPM Scratch (random init) — 10-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_scratch.json

# CDPM Scratch (random init) — 25-shot
python -m landmark_detection.landmarks_main -c configs/landmarks/hand/cdpm_scratch.json -n 25
```

---

## Checkpoints

The pretrained weights used in the paper are not published yet. They will be released on this page. Until then, train them with the commands above.

Each training script writes into its own run directory. The downstream configs expect the weights under `checkpoints/pretraining/`, so copy (or symlink) each output there, or point `model.pretrained_path` / `--pretrained_checkpoint` at the file directly.

| Pretraining run | File written by the script | Path the downstream config expects |
|---|---|---|
| Phase 1, `cdpm_combined.json` | `<save_dir>/combined/ckpt_45000_iterations.pt` | `checkpoints/pretraining/cdpm/combined_45k.pt` (input of Phase 2) |
| Phase 1, `cdpm_nih.json` | `<save_dir>/nih_unconditioned/ckpt_50000_iterations.pt` | `checkpoints/pretraining/cdpm/nih_50k.pt` |
| Phase 2, `cdpm_align.json` | `<save_dir>/final_5000.pt` | `checkpoints/pretraining/cdpm_align/lambda5.pt` |
| SSL, `ssl_<method>.json` | `<output_dir>/combined_landmarks/<method>_smp_resnet101/bs16x1_lr1e-04/encoder.pth` | `checkpoints/pretraining/ssl/<method>/encoder_only.pth` |
| DDPM (Di Via et al.), `ddpm_config.json --dataset <name>` | `<output_dir>/<name>/unet_t500_linear/bs1x8_lr1e-04/l2_ema/samples_all/models/last_ema_model.pt` | `checkpoints/pretraining/ddpm_divia/<name>.pt` |

`<save_dir>` and `<output_dir>` are the values in the pretraining config (`outputs/pretraining/...` by default). `<method>` is `dino`, `mocov3` or `simclrv2`; `<name>` is `chest`, `hand` or `cephalo`.
Phase 1 also writes `ckpt_<N>_iterations.pt` every 5k iterations and a rolling `last_model.pt`. Phase 2 also writes `ckpt_5000.pt` and `last_model.pt`, which hold the same weights as `final_5000.pt`.

The SSL `encoder.pth` can be used as it is. Its keys carry the `encoder.` prefix of the SSL backbone wrapper, and the downstream loader strips that prefix (and a `backbone.` prefix). No conversion to `encoder_only.pth` is needed: the configs keep that file name only so that all checkpoints follow one layout, and you can point them at `encoder.pth` instead. The loader stops with an error if no key in the file matches the encoder, and prints a warning that lists any missing keys.

`checkpoints/`, `*.pt` and `*.pth` are git-ignored.

---

## Datasets

The configs read all landmark datasets from one root, `data/`, relative to the repository root. To use another location, change `dataset.path` / `data_dir` in the configs.

```
data/
  chest/                  Shenzhen
    pngs/CHNCXR_*.png
    labels/CHNCXR_*.txt
  cephalo/                ISBI 2015
    jpg/*.jpg
    400_junior/*.txt
    400_senior/*.txt
  hand/                   Digital Hand Atlas
    jpg/*.jpg
    labels/all.csv
  NIH-Chest-Xray/         NIH ChestX-ray14 (CDPM (NIH) variant only)
    Data_Entry_2017.csv
    images_001/images/*.png ... images_012/images/*.png
```

| Dataset | Source | What the loader reads |
|---|---|---|
| Shenzhen (chest), 6 landmarks | Images: [Shenzhen Hospital CXR Set](https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/index.html) (also on [Kaggle](https://www.kaggle.com/nikhilpandey360/chest-xray-masks-and-labels)). Landmarks: [`chest_labels.zip`](https://github.com/MIRACLE-Center/YOLO_Universal_Anatomical_Landmark_Detection/blob/main/data/chest_labels.zip) from Zhu et al. | `pngs/<name>.png` and `labels/<name>.txt`. A label file has the number of landmarks on its first line, then one `x y` pair per line as fractions of the image width and height. Put only the images that have a label file in `pngs/`. |
| ISBI 2015 (cephalo), 19 landmarks | [figshare](https://figshare.com/s/37ec464af8e81ae6ebbf) or [Kaggle](https://www.kaggle.com/datasets/c34a0ef0cd3cfd5c5afbdb30f8541e887171f19f196b1ad63790ca5b28c0ec93) | `jpg/<name>.jpg` (the original `.bmp` images converted to JPEG), plus `400_junior/<name>.txt` and `400_senior/<name>.txt`, whose first 19 lines are `x,y` pixel coordinates. The target is the mean of the two annotators. Pixel spacing is 0.1 mm. |
| Digital Hand Atlas (hand), 37 landmarks | Images: [Digital Hand Atlas](https://ipilab.usc.edu/research/baaweb/). Landmarks: [`all.csv`](https://github.com/christianpayer/MedicalDataAugmentationTool-HeatmapRegression/blob/master/hand_xray/hand_xray_dataset/setup/all.csv) from Payer et al. | `jpg/<id>.jpg` with an integer file name, and `labels/all.csv` (no header): image id followed by 37 `x,y` pixel pairs. Landmark detection only uses images listed in `all.csv`. |
| NIH ChestX-ray14 | [NIH Clinical Center (Box)](https://nihcc.app.box.com/v/ChestXray-NIHCC) or [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data) | `Data_Entry_2017.csv` and `images_XXX/images/*.png`, images only. The loader splits patients 70/10/20 % into train/val/test (seed 42); pretraining uses train+val. |

**Splits.** The loaders in `datasets/*_dataset.py` sort the files by name and split them deterministically:

| Dataset | Train | Val | Test |
|---|---|---|---|
| Shenzhen | first 195 | next 34 | remaining 50 |
| ISBI 2015 | first 130 | next 20 | remaining 250 |
| DHA | first 550 | next 59 | all remaining labelled images |

The few-shot training set (`-n 10` / `-n 25`) is the first N images of the train split, so it is the same for every seed. Its file list is saved as `train_image_files.json` in the run's output directory.
The unlabeled pretraining corpus for CDPM, CDPM-Align and the SSL baselines is train+val of the three datasets: 229 + 150 + 609 = 988 images. Test images are never used for pretraining. Each Di Via et al. DDPM is pretrained on train+val of its own dataset.

**DHA millimetre conversion.** The paper (Sec. 2.3) says DHA errors are converted to mm with the DICOM pixel spacing. The code does not read a pixel spacing. Instead, `datasets/hand_dataset.py` assumes a wrist width of 50 mm and sets each image's mm-per-pixel to 50 divided by the pixel distance between landmarks 1 and 5. DHA errors in mm reported by this code depend on that assumption.

| Dataset | Paper name | Landmarks | Default sigma |
|---|---|---|---|
| chest | Shenzhen | 6 | 5 |
| hand | DHA | 37 | 5 |
| cephalo | ISBI2015 | 19 | 1 |

---

## Configs

```
configs/
  pretraining/
    cdpm_combined.json       — CDPM Phase 1, combined dataset
    cdpm_nih.json            — CDPM Phase 1, NIH ChestX-ray14
    cdpm_align.json          — CDPM-Align Phase 2 fine-tuning
    ssl_dino.json            — DINO SSL
    ssl_mocov3.json          — MoCo v3 SSL
    ssl_simclrv2.json        — SimCLR v2 SSL
    ddpm_config.json         — DDPM DiVia (dataset overridden per run)
  landmarks/
    chest/     cdpm_align.json  cdpm_nih.json  cdpm_scratch.json  ddpm_divia.json
               imagenet.json  ssl_dino.json  ssl_mocov3.json  ssl_simclrv2.json
    hand/      (same 8 files)
    cephalo/   (same 8 files)
```

### λ ablation

The λ ablation (paper Table 4, Shenzhen) is not packaged as configs. Rerun Phase 2 from the same Phase 1 checkpoint with a different `--lambda_align`, and give each λ its own `--save_dir`. Otherwise every run writes `final_5000.pt` to the same directory and overwrites the previous one:

```bash
for lam in 0 1 3 5 10; do
    python -m pretraining.cdpm_align.run_finetune -c configs/pretraining/cdpm_align.json \
        --lambda_align $lam \
        --pretrained_checkpoint checkpoints/pretraining/cdpm/combined_45k.pt \
        --save_dir outputs/pretraining/cdpm_align/lambda$lam
done
```

Then run the chest experiments with `model.pretrained_path` set to each `final_5000.pt`. Use one copy of `configs/landmarks/chest/cdpm_align.json` per λ with its own file name (for example `cdpm_align_lambda0.json`), because the output directory is named after the config file.

The paper describes the λ=0 row only as "without alignment (λ=0)" and does not say how it was trained. The loop above reproduces it as the same 5k-iteration Phase 2 fine-tuning with `--lambda_align 0`, i.e. with the diffusion loss only. This row is not the same model as the "CDPM (NIH, λ=0)" rows of Tables 1–3, which come from NIH pretraining.

---

## Requirements

Python 3.10.

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@inproceedings{divia2026cdpm,
  title     = {{CDPM-Align}: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection},
  author    = {Di Via, Roberto and Voiculescu, Irina and Odone, Francesca and Pastore, Vito Paolo},
  booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2026},
  series    = {Lecture Notes in Computer Science},
  volume    = {16881},
  pages     = {89--99},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  doi       = {10.1007/978-3-032-38072-2_9}
}
```
