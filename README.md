# DynaMoFE: Dynamic Mixture of Forensic Experts for Codec-Resilient Deepfake Video Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Format: IEEE](https://img.shields.io/badge/Paper-IEEE%20Format-blue.svg)](paper/paper.pdf)

Official implementation, evaluation suite, and technical artifacts for **DynaMoFE** (*Dynamic Mixture of Forensic Experts with Degradation-Aware Routing for Codec-Resilient Deepfake Video Detection*).

---

## 📌 Executive Summary

### The Forensic Trade-Off Dilemma
Deepfake video detectors trained on clean benchmarks experience severe performance degradation when deployed across lossy transmission channels, streaming services, and social media codecs (JPEG, WebP, H.264, H.265). Existing models face an intrinsic domain trade-off:
* **Semantic Component Models (e.g., FCG):** Excel on clean video and cross-dataset transfer via foundation ViT priors, but drop sharply under heavy compression (-8.41 pp under H.264).
* **Spatiotemporal Thumbnail Networks (e.g., TALL):** Invariant to face crop provenance (+0.037 pp), but thumbnail downsampling destroys subtle high-frequency boundaries, causing severe collapse under high compression (76.40% on H.264 CRF 35).
* **Frequency-Domain Networks (e.g., F3Net):** Learnable DCT filters isolate generative grid discrepancies surviving compression, but lack high-level anatomical consistency.

### The DynaMoFE Solution
**DynaMoFE** resolves this trade-off by conditioning forensic routing on **physical transmission degradation signatures** without requiring forgery labels:
1. **Physical Degradation Signature Extractor:** Extracts a 16-D deterministic vector capturing 2D-FFT spectral roll-off, $8\times8$ block boundary discontinuities, inter-frame temporal motion entropy, and total variation norm.
2. **Dynamic Gating Neural Router:** Processes the degradation signature to predict instance-level mixture weights across heterogeneous forensic experts (semantic component, spatiotemporal thumbnail, frequency decomposition).
3. **Adaptive Decision Fusion:** Dynamically shifts reliance toward frequency and spatiotemporal branches under heavy compression, while prioritizing facial component guidance on pristine video.

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph Input ["Input Video Stream"]
        V["Raw Video X = (f_1, ..., f_T)"]
    end

    subgraph FeatureExtraction ["Parallel Processing Engine"]
        DE["Physical Degradation Extractor D(X)"]
        
        subgraph Experts ["Heterogeneous Forensic Experts"]
            E1["FCG: Semantic Component Guidance (CLIP ViT-L/14)"]
            E2["TALL: Spatiotemporal Thumbnail (Swin-B)"]
            E3["F3Net: Frequency DCT Decomposition (FAD + LFS)"]
            E4["Xception: Spatial Boundary Baseline"]
        end
    end

    subgraph DynamicRouting ["Dynamic Gating Subsystem"]
        D_SIG["16-D Degradation Signature d(X)"]
        ROUTER["Dynamic Neural Router R_theta\n(LayerNorm + GELU + Softmax)"]
        WEIGHTS["Dynamic Routing Weights w(X) in Delta^(M-1)"]
    end

    subgraph Standardization ["Margin Standardization"]
        S_RAW["Raw Decision Margins: [s_fcg, s_tall, s_f3net, s_xcep]"]
        S_NORM["Standardized Margins: s_tilde_m = (s_m - mu_m) / sigma_m"]
    end

    subgraph DecisionFusion ["Adaptive Decision Head"]
        FUSION["Weighted Aggregation: m_DynaMoFE = sum w_m * s_tilde_m"]
        DECISION["Final Authenticity Score & Classification"]
    end

    V --> DE
    V --> E1
    V --> E2
    V --> E3
    V --> E4

    DE --> D_SIG
    D_SIG --> ROUTER
    ROUTER --> WEIGHTS

    E1 --> S_RAW
    E2 --> S_RAW
    E3 --> S_RAW
    E4 --> S_RAW

    S_RAW --> S_NORM
    WEIGHTS --> FUSION
    S_NORM --> FUSION
    FUSION --> DECISION
```

---

## 📊 Benchmark Results

### Matched Codec Robustness Benchmark (FaceForensics++ C23 Test, 700 Videos, 70 Source Clusters)

| Method | Clean | JPEG-40 | WebP-50 | H.264 (CRF 35) | H.265 (CRF 32) | Res $\rightarrow$ H.264 | H.264 $\rightarrow$ Res | Sealed Mean | Worst Codec | Retention |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TALL (Local Seed 42)** | 98.38 | 89.74 | 92.47 | 76.40 | 84.82 | 90.27 | 91.33 | 87.50 | 76.40 | 77.5% |
| **ForensicsAdapter** | 95.60 | 88.42 | 88.93 | 86.10 | 85.92 | 86.74 | 88.31 | 87.40 | 85.92 | 82.0% |
| **Xception** | 98.55 | 93.37 | 91.33 | 87.88 | 88.77 | 90.34 | 93.12 | 90.80 | 87.88 | 84.0% |
| **F3Net** | 98.60 | 93.87 | 92.21 | 87.35 | 90.58 | 90.70 | 92.80 | 91.25 | 87.35 | 84.9% |
| **FCG (Official CVPR 2025)** | 98.66 | 96.96 | 94.74 | 90.25 | 91.55 | 91.75 | 93.55 | 93.14 | 90.25 | 88.7% |
| **$\text{DynaMoFE}_{\text{static}}$ (Tri-Domain)** | 99.38 | 98.24 | 96.49 | 92.83 | 94.70 | 95.37 | 96.60 | 95.71 | 92.83 | 92.6% |
| **$\text{DynaMoFE}_{\text{static}}$ (Quad-Domain)** | **99.35** | **98.24** | **96.31** | **93.24** | **94.71** | **95.52** | **96.87** | **95.81** | **93.24** | **92.8%** |
| **$\text{DynaMoFE}_{\text{adaptive}}$ (OOF Gating)** | 99.30 | 97.12 | 94.73 | 91.15 | 92.82 | 94.70 | 96.06 | 94.43 | 91.15 | 90.1% |

### Key Empirical Highlights:
* **Multi-Domain Forensic Synergy SOTA:** $\text{DynaMoFE}_{\text{static}}$ reaches **95.81% sealed mean AUC** (+2.67 pp over FCG 93.14%, +4.56 pp over F3Net, +8.31 pp over TALL).
* **Worst-Case Robustness (H.264 CRF 35):** $\text{DynaMoFE}_{\text{static}}$ reaches **93.24% AUC** (+2.99 pp over FCG 90.25%, +16.84 pp over TALL 76.40%). $\text{DynaMoFE}_{\text{adaptive}}$ reaches **91.15% AUC** (+0.90 pp over FCG, +14.75 pp over TALL).
* **Cross-Dataset Transfer (Celeb-DF-v2):** Achieves **93.84% AUC**, outperforming standalone TALL (84.92%) by **+8.92 pp** and FCG (93.58%).
* **Exact 10,000-Replicate Paired Cluster Bootstrap Significance:**
  * $\text{DynaMoFE}_{\text{static}}$ achieves strictly positive $95\%$ confidence intervals against all external baselines across all environments ($p < 0.05$ on clean; $p < 0.001$ on H.264).
  * $\text{DynaMoFE}_{\text{adaptive}}$ achieves strictly positive confidence intervals under compression against all external models ($p < 0.001$). On clean canonical video, it significantly outperforms TALL ($+0.92$ pp), F3Net ($+0.69$ pp), Xception ($+0.74$ pp), and ForensicsAdapter ($+3.69$ pp) with $p < 0.05$, while against the high-capacity FCG foundation model on clean video, the margin is $+0.64$ pp ($95\%$ CI: $[-0.03, +1.29]$, $p = 0.058$).
* **Analytical Static vs. Adaptive Dynamics:** While the theoretical oracle upper bound across environments is 95.93% (+0.12 pp over static), learning a 16-parameter neural router across 70 source identity clusters introduces sample-level estimation variance that slightly offsets this headroom. A fixed multi-domain prior therefore delivers zero estimation variance and superior out-of-fold generalization.
* **Relation to 2026 Literature:** Evaluated alongside 2026 advances including TriMoE (CVPRW 2026), WGN (CVPRW 2026), UMCL (IJCV 2026), GenD (CVPR 2026), and QTFP (2026). In contrast to TriMoE's latent feature routing, DynaMoFE routes via non-semantic physical degradation signatures.

---

## 📁 Repository Structure

```
DynaMoFE/
├── README.md                              # This file
├── requirements.txt                       # Package dependencies
├── setup.py                               # Package installation script
├── LICENSE                                # MIT License
│
├── dynamofe/                              # Core Python module
│   ├── __init__.py
│   ├── degradation.py                     # 16-D Physical Degradation Signature Extractor
│   ├── router.py                          # DynamicGatingRouter & DynaMoFEDetector
│   ├── extract_degradations.py            # Feature extraction script
│   ├── train_router.py                    # 5-fold cluster cross-validation training
│   ├── evaluate_dynamofe.py               # Benchmarking & 10,000 bootstrap significance engine
│   ├── analyze_routing.py                 # Weight visualization and figure generation
│   └── test_dynamofe.py                   # Component unit tests
│
├── data/                                  # Pre-extracted data & predictions for reproduction
│   ├── degradations/                      # FF++ and Celeb-DF degradation vectors (.pt)
│   └── predictions/                       # Baseline predictions (FCG, TALL, F3Net, Xception, FA)
│
├── outputs/                               # Results, metrics, and generated figures
│   ├── dynamofe_results.json              # Full per-video prediction results
│   ├── dynamofe_benchmark_metrics.csv     # AUC, AP, EER across environments
│   ├── dynamofe_main_comparison_auc.csv   # Side-by-side model comparison table
│   ├── dynamofe_paired_bootstraps.csv     # 10,000 bootstrap CI table
│   └── figures/                           # Publication figures (PDF and PNG)
│
├── paper/                                 # Complete IEEE Conference Submission Package
│   ├── paper.tex                          # LaTeX manuscript (IEEEtran conference format)
│   ├── references.bib                     # Complete BibTeX citations
│   ├── paper.pdf                          # Compiled 5-page PDF manuscript
│   ├── IEEEtran.cls / .bst                # Official IEEE conference formatting classes
│   └── figures/                           # High-res vector graphics
│
├── patent/                                # Intellectual Property Documentation
│   └── PATENT_APPLICATION_DYNAMOFE.md     # Full patent disclosure with 20 formal claims
│
└── docs/                                  # Technical Specifications
    └── DYNAMOFE_MODEL_AND_ARCHITECTURE.md # Exhaustive mathematical and architectural spec
```

---

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/your-org/DynaMoFE.git
cd DynaMoFE
pip install -e .
```

### 2. Run Unit Tests

```bash
python -c "
import dynamofe.test_dynamofe as t
t.test_degradation_extractor()
t.test_router_forward_and_gradients()
t.test_dynamofe_detector()
print('All unit tests passed!')
"
```

### 3. Reproduce Training & Evaluation

```bash
# Train dynamic router across 5 source-cluster cross-validation folds
python dynamofe/train_router.py

# Evaluate matched benchmarks and run 10,000 paired cluster bootstraps
python dynamofe/evaluate_dynamofe.py

# Generate publication-grade figures
python dynamofe/analyze_routing.py
```

### 4. Compile Paper Manuscript

```bash
cd paper
latexmk -pdf paper.tex
```

---

## 📄 Citation

```bibtex
@inproceedings{dynamofe2026,
  title     = {DynaMoFE: Dynamic Mixture of Forensic Experts with Degradation-Aware Routing for Codec-Resilient Deepfake Video Detection},
  author    = {Anonymous Author(s)},
  booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

---

## ⚖️ License & Patent Notice

* The source code is released under the **MIT License**.
* The architectural mechanisms, degradation signature extractor, and dynamic gating subsystem described herein are subject to intellectual property protection. See [`patent/PATENT_APPLICATION_DYNAMOFE.md`](patent/PATENT_APPLICATION_DYNAMOFE.md) for full patent application details.
