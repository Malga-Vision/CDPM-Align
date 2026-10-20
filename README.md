# CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection

[![Paper](https://img.shields.io/badge/Paper-MICCAI%202026-blue)](https://papers.miccai.org/miccai-2026/paper/3797_paper.pdf)
[![Project page](https://img.shields.io/badge/Project-Page-green)](https://malga-vision.github.io/CDPM-Align_AcademicPage/)

**Roberto Di Via, Irina Voiculescu, Francesca Odone, Vito Paolo Pastore**
MaLGa, DIBRIS, University of Genoa, Italy · Department of Computer Science, University of Oxford, UK

Official code for the MICCAI 2026 paper **"CDPM-Align: Multi-Scale Guidance-Aligned Diffusion Pretraining for Robust Few-Shot Anatomical Landmark Detection"**.

[Paper (PDF)](https://papers.miccai.org/miccai-2026/paper/3797_paper.pdf) · [Project page](https://malga-vision.github.io/CDPM-Align_AcademicPage/)

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

---

## Pretraining

### Phase 1 — CDPM base (combined chest + hand + cephalo, 50k iterations, batch size 16)

```bash
python -m pretraining.cdpm.run -c configs/pretraining/cdpm_combined.json
```

### Phase 1 — CDPM base (NIH ChestX-ray14, 50k iterations)

```bash
python -m pretraining.cdpm.run -c configs/pretraining/cdpm_nih.json
```

### Phase 2 — CDPM-Align fine-tuning (5k iterations, λ=5, batch size 4, lr 1e-5)

```bash
python -m pretraining.cdpm_align.run_finetune \
    -c configs/pretraining/cdpm_align.json \
    --lambda_align 5.0 \
    --pretrained_checkpoint checkpoints/pretraining/cdpm/combined_50k.pt
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

All downstream experiments use the NLL loss, AdamW (lr=1e-4), an effective batch size of 8, up to 300 epochs, and early stopping (patience=15).
Default: **10 training samples**. Override with `-n N` (e.g., `-n 25`).

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

Best experiment checkpoints are symlinked under `checkpoints/pretraining/`:

```
checkpoints/pretraining/
  cdpm/combined_50k.pt          — CDPM base, 50k iterations, combined dataset (chest+hand+cephalo)
  cdpm/nih_50k.pt               — CDPM base, 50k iterations, NIH ChestX-ray14
  cdpm_align/lambda5.pt         — CDPM-Align, λ=5, the checkpoint used in the paper (converted, Config-free)
  ssl/dino/encoder_only.pth     — DINO ResNet-101 encoder
  ssl/mocov3/encoder_only.pth   — MoCo v3 ResNet-101 encoder
  ssl/simclrv2/encoder_only.pth — SimCLR v2 ResNet-101 encoder
  ddpm_divia/chest.pt           — DiVia DDPM, Shenzhen domain
  ddpm_divia/hand.pt            — DiVia DDPM, DHA domain
  ddpm_divia/cephalo.pt         — DiVia DDPM, ISBI2015 domain
```

---

## Datasets

Data root: `/nethome/home/user/Datasets/Landmarks_datasets/`

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

### Notes on the paper text

The configs reproduce the runs behind the published tables. Two settings differ from how the paper words them:

- CDPM-Align starts from the 50k-iteration CDPM base (`combined_50k.pt`), followed by 5k alignment iterations at batch size 4. The paper describes this as "50k iterations (45k standard diffusion, 5k alignment fine-tuning)".
- Downstream training allows up to 300 epochs, with early stopping (patience 15). The paper says 200 epochs.

The λ ablation (paper Table 4) is not packaged as configs; rerun Phase 2 with a different `--lambda_align` to reproduce it.

---

## Requirements

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
  publisher = {Springer Nature Switzerland},
  year      = {2026}
}
```
