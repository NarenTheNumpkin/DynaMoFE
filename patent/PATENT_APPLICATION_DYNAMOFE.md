# PATENT APPLICATION SPECIFICATION

**DOCKET NO.:** DYN-MOFE-2026-001  
**FILING DATE:** September 19, 2026  
**TITLE:** METHOD, SYSTEM, AND COMPUTER-READABLE MEDIUM FOR DEEPFAKE VIDEO DETECTION VIA DYNAMIC MULTI-DOMAIN FORENSIC ROUTING CONDITIONED ON TRANSMISSION DEGRADATION SIGNATURES

---

## ABSTRACT

Methods, systems, and non-transitory computer-readable media for robust deepfake video detection under lossy transmission channels and heterogeneous video manipulations are disclosed. A computer-implemented video inspection system extracts a multi-dimensional physical degradation signature from an input video sequence without relying on semantic labels, capturing spectral power roll-off, block boundary discontinuities, inter-frame temporal motion entropy, and spatial gradient distribution. The input video is concurrently processed through a plurality of specialized forensic expert branches comprising: (i) a facial-component-guided semantic branch modeling fine-grained anatomical landmark consistency; (ii) a spatiotemporal thumbnail branch modeling cross-frame continuity in a compact layout; and (iii) a frequency-domain branch modeling periodic generative synthesis and compression artifacts via discrete cosine transform (DCT) decomposition. A dynamic neural router processes the physical degradation signature to predict instance-level routing weights across the expert branches. The router dynamically modulates expert reliance based on channel-induced perturbations, shifting weight toward frequency and spatiotemporal branches under heavy recompression and downscaling, and toward semantic component guidance under high-fidelity inputs. The system aggregates standardized expert decision margins using the predicted weights to produce a robust, transmission-resilient authenticity decision margin.

---

## 1. FIELD OF THE INVENTION

The present disclosure relates generally to computer vision, digital forensics, and deep learning. More specifically, the present disclosure relates to methods, systems, and computer-readable media for detecting manipulated, synthetic, and hyper-realistic facial videos (colloquially termed "deepfakes") across lossy social-media transmission channels, heterogeneous video compression codecs, and unseen manipulation pipelines.

---

## 2. BACKGROUND OF THE INVENTION

### 2.1 Description of Related Art
Deepfake video generation techniques—including generative adversarial networks (GANs), diffusion models, autoencoders, and facial neural radiance fields (NeRFs)—have evolved to synthesize hyper-realistic human facial videos with high perceptual plausibility. In response, numerous deepfake detection algorithms have been proposed. Existing detectors generally categorize into three distinct paradigms:

1. **Spatial and Semantic Facial Detectors:** Methods that inspect individual frames or high-level facial semantics, such as facial landmark consistency, facial component interactions (e.g., eyes, lips, nose, skin), or foundation model features (e.g., CLIP ViT representations). While showing strong cross-dataset generalization on uncompressed or high-quality video, these architectures are vulnerable to spatial recompression, boundary cropping shifts, and high-frequency loss.
2. **Spatiotemporal Video Detectors:** Methods that model temporal discrepancies across consecutive video frames, such as thumbnail layouts (e.g., TALL) or 3D convolutional networks. While spatiotemporal networks capture multi-frame coherence and exhibit robustness to bounding-box spatial jitters, they frequently overfit to dataset-specific spatial textures and struggle when transferred to distinct generative manipulation families.
3. **Frequency-Domain Forensic Detectors:** Methods that transform facial images into frequency representations (e.g., Discrete Cosine Transform [DCT] or Fast Fourier Transform [FFT]) to capture high-frequency synthetic artifacts and grid irregularities. While frequency-based detectors capture compression-resistant periodic signatures, they lack anatomical and semantic reasoning, leading to false alarms when encountering natural photographic high-frequency details.

### 2.2 Shortcomings of the Prior Art and the "Forensic Dilemma"
In real-world deployment (e.g., social media platforms, communication networks, and legal provenance authentication), video streams rarely remain pristine. Videos are subjected to aggressive lossy transmission channels, including:
- High-quantization block-based compression (e.g., JPEG with quality factor $Q \le 40$, WebP with quality factor $Q \le 50$);
- Advanced predictive inter-frame video compression (e.g., H.264/AVC with Constant Rate Factor $\text{CRF} \ge 35$, H.265/HEVC with $\text{CRF} \ge 32$);
- Multi-stage transcoding and spatial downscaling pipelines (e.g., resizing prior to video encoding, or re-encoding after display scaling).

Empirical evaluations reveal that every single-domain detector suffers severe performance degradation under such real-world channel perturbations. For example, state-of-the-art facial component models drop by over 8 percentage points in Area Under the ROC Curve (AUC) under H.264 compression, while thumbnail models drop by more than 21 percentage points under heavy bit-rate constraints.

Conventional ensemble methods (such as fixed-weight linear averaging or simple majority voting) fail to resolve this vulnerability because fixed weights cannot adapt to the severity or nature of video degradation. An ensemble that heavily weights a spatial component branch will suffer catastrophic failure when presented with an H.264-compressed video where high-frequency facial component boundaries have been obliterated by quantization.

Accordingly, there is an urgent and unmet technological need for an adaptive deepfake detection framework that dynamically identifies transmission channel degradations in real time and automatically routes reliance among complementary forensic domains to maximize detection accuracy and robustness.

---

## 3. SUMMARY OF THE INVENTION

To overcome the deficiencies of the prior art, the present disclosure provides a novel system and method termed **DynaMoFE (Dynamic Mixture of Forensic Experts)**.

In a first aspect, the present invention provides a computer-implemented method for deepfake video detection under arbitrary transmission channel perturbations, comprising:
1. Receiving an input video comprising a sequence of facial image frames;
2. Extracting, via a deterministic degradation extractor, a $K$-dimensional physical degradation signature characterizing non-semantic transmission channel artifacts of the input video;
3. Concurrently executing a plurality of $M$ heterogeneous forensic expert branches on the input video, wherein each expert branch processes the input video within a distinct forensic domain and outputs an expert decision margin;
4. Processing the physical degradation signature through a dynamic neural router to compute instance-level mixture weights across the $M$ forensic expert branches;
5. Standardizing each expert decision margin utilizing pre-calibrated baseline domain statistics; and
6. Fusing the standardized expert decision margins in accordance with the predicted dynamic mixture weights to generate an authenticity decision margin indicating whether the input video is authentic or forged.

In a second aspect, the physical degradation signature comprises a set of orthogonal signal metrics computed directly from pixel data without requiring class label supervision:
- High-frequency power spectral ratio ($\text{HF}_{\text{ratio}}$) and mid-frequency spectral ratio ($\text{MF}_{\text{ratio}}$) computed via 2D Fast Fourier Transform (2D-FFT);
- Power spectral decay slope ($\beta_{\text{decay}}$) across logarithmic radial frequency rings;
- Horizontal and vertical block boundary discontinuity ratios ($B_h, B_v$) measuring 8×8 pixel grid step discontinuities characteristic of block-based compression;
- Inter-frame temporal difference magnitude ($\mu_{\Delta}$), temporal variance ($\sigma_{\Delta}^2$), and peak inter-frame change ($\max \Delta$);
- Spatial total variation (TV) norm and gradient magnitude distribution statistics; and
- Luminance dynamic range, contrast, and shadow/highlight clipping fractions.

In a third aspect, the plurality of $M$ forensic expert branches comprises at least:
- An anatomical facial component guided foundation model (FCG) utilizing a frozen Vision Transformer (CLIP ViT-L/14) and learned spatial component queries;
- A spatiotemporal thumbnail layout network (TALL) processing multi-frame thumbnails via a Swin Transformer backbone; and
- A dual-stream frequency-aware decomposition network (F3Net) utilizing learnable DCT frequency bandpass filters and local frequency statistics.

In a fourth aspect, the dynamic neural router is trained using a multi-objective loss comprising a classification margin loss (binary cross-entropy) and a Shannon entropy diversity regularizer that penalizes expert collapse, ensuring balanced, degradation-dependent routing across all operating regimes.

---

## 4. BRIEF DESCRIPTION OF THE DRAWINGS

- **FIG. 1** is a high-level block diagram illustrating the end-to-end architecture of the DynaMoFE deepfake video detection system.
- **FIG. 2** is a schematic diagram illustrating the Physical Degradation Signature Extractor and its mathematical feature components.
- **FIG. 3** is a schematic diagram illustrating the internal architecture of the Dynamic Gating Router.
- **FIG. 4** is a chart illustrating the dynamic routing weight allocation across clean canonical video and six sealed compression codec environments.
- **FIG. 5** is a comparative performance graph showing ROC-AUC improvements of DynaMoFE over individual state-of-the-art baselines across multiple transmission codecs.
- **FIG. 6** is a block diagram of an exemplary computing system configured to implement the DynaMoFE method.

---

## 5. DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS

Referring to **FIG. 1**, the DynaMoFE architecture operates on an incoming digital video stream comprising $T$ facial frames $X = (f_1, f_2, \ldots, f_T)$. The video is simultaneously delivered to:
1. The **Degradation Signature Extractor 110**; and
2. The **Forensic Expert Ensemble 120** comprising Expert 1 (Semantic Component FCG 122), Expert 2 (Spatiotemporal Thumbnail TALL 124), Expert 3 (Frequency Decomposition F3Net 126), and optionally Expert 4 (Spatial Texture Xception 128).

### 5.1 Physical Degradation Signature Extraction
Referring to **FIG. 2**, the Degradation Signature Extractor 110 processes the luminance channel $Y$ of the facial crops resized to a standardized dimension $S \times S$ (preferably $S=128$). The extractor computes a $K$-dimensional vector $\mathbf{d}(X) = [d_1, d_2, \ldots, d_{16}]^T$ as follows:

1. **Spectral Power Roll-off:** The 2D Discrete Fourier Transform of luminance $Y$ is computed as:
   $$\mathcal{F}(u, v) = \sum_{x=0}^{S-1} \sum_{y=0}^{S-1} Y(x, y) e^{-j 2\pi (\frac{ux}{S} + \frac{vy}{S})}$$
   The centered power spectrum is $P(u, v) = |\mathcal{F}_{\text{shift}}(u, v)|^2$. The normalized radial distance from DC center $(S/2, S/2)$ is:
   $$r(u, v) = \frac{\sqrt{(u - S/2)^2 + (v - S/2)^2}}{(S/2)\sqrt{2}} \in [0, 1]$$
   The high-frequency energy ratio $d_1$ and mid-frequency ratio $d_2$ are defined by:
   $$d_1 = \frac{\sum_{r(u, v) \ge 0.6} P(u, v)}{\sum_{u, v} P(u, v) + \epsilon}, \quad d_2 = \frac{\sum_{0.25 \le r(u, v) < 0.6} P(u, v)}{\sum_{u, v} P(u, v) + \epsilon}$$
   The spectral decay slope $d_3$ is the linear regression slope of $\log(P_k)$ across 4 radial frequency octaves.

2. **Block Boundary Discontinuity (Blockiness Metric):** Block-based transforms (e.g., JPEG, MPEG-2/4, H.264) introduce artificial grid discontinuities at multiples of 8 pixels. Let $\Delta_h(x, y) = |Y(x+1, y) - Y(x, y)|$. The boundary jump $J_{\text{bound}}$ and internal jump $J_{\text{int}}$ are:
   $$J_{\text{bound}} = \frac{1}{|\Omega_B|} \sum_{x \in \{7, 15, \ldots, S-1\}} \Delta_h(x, y), \quad J_{\text{int}} = \frac{1}{|\Omega_I|} \sum_{x \in \{3, 11, \ldots, S-1\}} \Delta_h(x, y)$$
   The horizontal and vertical blockiness ratios are:
   $$d_4 = \frac{J_{\text{bound}, h}}{J_{\text{int}, h} + \epsilon}, \quad d_5 = \frac{J_{\text{bound}, v}}{J_{\text{int}, v} + \epsilon}, \quad d_6 = \frac{1}{2}(d_4 + d_5)$$

3. **Inter-Frame Temporal Dynamics:** For multi-frame observations ($T > 1$):
   $$\Delta_t = \frac{1}{HW} \sum_{x, y} |f_{t+1}(x, y) - f_t(x, y)|$$
   The mean motion $d_7 = \frac{1}{T-1}\sum \Delta_t$, variance $d_8 = \text{Var}(\Delta_t)$, and peak motion $d_9 = \max_t \Delta_t$.

4. **Spatial Gradient and Dynamic Range:** Spatial Total Variation (TV) norm:
   $$d_{10} = \frac{1}{HW} \sum_{x, y} (|\nabla_x Y| + |\nabla_y Y|)$$
   Gradient magnitude mean $d_{11}$ and std $d_{12}$; luminance mean $d_{13}$ and contrast std $d_{14}$; crushed shadow fraction $d_{15} = \frac{1}{HW}\sum \mathbb{I}(Y \le 3/255)$ and blown highlight fraction $d_{16} = \frac{1}{HW}\sum \mathbb{I}(Y \ge 252/255)$.

### 5.2 Dynamic Neural Router Architecture
Referring to **FIG. 3**, the Dynamic Neural Router $\mathcal{R}_\theta$ comprises a multi-layer perceptron:
$$\mathbf{h}_1 = \text{GELU}(\text{LayerNorm}(W_1 \mathbf{d}(X) + \mathbf{b}_1))$$
$$\mathbf{h}_2 = \text{GELU}(\text{LayerNorm}(W_2 \mathbf{h}_1 + \mathbf{b}_2))$$
$$\mathbf{z}_{\text{gate}} = W_3 \mathbf{h}_2 + \mathbf{b}_3 \in \mathbb{R}^M$$
The routing weights $\mathbf{w}(X) = [w_1, \ldots, w_M]^T$ are computed via temperature-scaled Softmax:
$$w_m(X) = \frac{\exp(z_{\text{gate}, m} / \tau)}{\sum_{j=1}^M \exp(z_{\text{gate}, j} / \tau)}, \quad \sum_{m=1}^M w_m(X) = 1, \quad w_m(X) \ge 0$$

### 5.3 Standardization and Adaptive Decision Fusion
Each forensic expert $m \in \{1, \ldots, M\}$ outputs an uncalibrated decision margin $s_m(X) \in \mathbb{R}$. To prevent dominant magnitude bias, each margin is standardized using pre-computed validation statistics:
$$\tilde{s}_m(X) = \frac{s_m(X) - \mu_m}{\sigma_m + \epsilon}$$
where $\mu_m = \mathbb{E}[s_m]$ and $\sigma_m = \sqrt{\text{Var}(s_m)}$ on canonical validation data.
The final fused decision margin $m_{\text{DynaMoFE}}(X)$ is:
$$m_{\text{DynaMoFE}}(X) = \sum_{m=1}^M w_m(X) \cdot \tilde{s}_m(X)$$
The posterior probability of manipulation is computed via sigmoid: $P(\text{Fake} \mid X) = \sigma(m_{\text{DynaMoFE}}(X))$.

### 5.4 Multi-Objective Training and Diversity Regularization
The router parameters $\theta$ are trained using a joint objective:
$$\mathcal{L}(\theta) = \mathcal{L}_{\text{BCE}}(m_{\text{DynaMoFE}}(X), y) + \lambda_{\text{div}} \mathcal{L}_{\text{div}}(\mathbf{w})$$
where $\mathcal{L}_{\text{BCE}}$ is Binary Cross-Entropy with Logits, and $\mathcal{L}_{\text{div}}$ penalizes expert collapse by maximizing the Shannon entropy of the batch-averaged routing weights:
$$\bar{\mathbf{w}} = \frac{1}{B} \sum_{i=1}^B \mathbf{w}(X_i)$$
$$\mathcal{L}_{\text{div}}(\mathbf{w}) = \log(M) - \mathcal{H}(\bar{\mathbf{w}}) = \log(M) + \sum_{m=1}^M \bar{w}_m \log(\bar{w}_m)$$
Minimizing $\mathcal{L}_{\text{div}}$ forces the router to maintain active representation across all forensic modalities while freely specializing per instance.

---

## 6. EXPERIMENTAL RESULTS AND PROVABLE TECHNICAL ADVANTAGES

The DynaMoFE system was rigorously evaluated in a matched benchmark against all primary state-of-the-art baselines on 700 FaceForensics++ ($C23$) test videos across seven deterministic environments (canonical clean, JPEG-40, WebP-50, H.264 CRF-35, H.265 CRF-32, Resize$\rightarrow$H.264, and H.264$\rightarrow$Resize).

### Matched Benchmark Results (ROC-AUC %)
| Model | Canonical | JPEG-40 | WebP-50 | H.264 (CRF 35) | H.265 (CRF 32) | Resize$\rightarrow$H264 | H264$\rightarrow$Resize | Sealed Codec Mean |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TALL (Local) | 98.38 | 89.74 | 92.47 | 76.40 | 84.82 | 90.27 | 91.33 | 87.50 |
| ForensicsAdapter | 95.60 | 88.42 | 88.93 | 86.10 | 85.92 | 86.74 | 88.31 | 87.40 |
| Xception | 98.55 | 93.37 | 91.33 | 87.88 | 88.77 | 90.34 | 93.12 | 90.80 |
| F3Net | 98.60 | 93.87 | 92.21 | 87.35 | 90.58 | 90.70 | 92.80 | 91.25 |
| FCG (Official SOTA) | 98.66 | 96.96 | 94.74 | 90.25 | 91.55 | 91.75 | 93.55 | 93.14 |
| **DynaMoFE (Ours)** | **99.38** | **98.24** | **96.49** | **92.83** | **94.70** | **95.37** | **96.60** | **95.71** |

DynaMoFE outperforms every external detector across every single compression environment. Crucially, under the most severe degradation (H.264 CRF 35), DynaMoFE improves upon the strongest prior model (FCG) by **+2.58 percentage points** and upon TALL by **+16.43 percentage points**. In 10,000-replicate paired cluster bootstrap hypothesis tests, the 95% confidence intervals against all five external models are strictly positive.

---

## 7. PATENT CLAIMS

What is claimed is:

1. A computer-implemented method for detecting deepfake videos under lossy transmission channel conditions, the method comprising:
   - receiving an input video comprising a sequence of facial image frames;
   - extracting, using an automated degradation extraction module, a multi-dimensional physical degradation signature from the input video, wherein the physical degradation signature quantifies signal perturbations of the input video without utilizing semantic facial forgery labels;
   - executing a plurality of heterogeneous forensic expert neural network branches on the input video, wherein each expert branch models forgery traces in a distinct forensic domain and outputs an expert decision score;
   - generating, via a dynamic neural router conditioned on the physical degradation signature, a set of dynamic routing weights corresponding to the plurality of forensic expert branches;
   - standardizing each expert decision score using domain-specific baseline distribution parameters; and
   - computing an authenticity decision margin by fusing the standardized expert decision scores in accordance with the dynamic routing weights, wherein the authenticity decision margin indicates whether the input video is authentic or manipulated.

2. The method of claim 1, wherein the physical degradation signature comprises at least a high-frequency power spectral decay metric computed via a two-dimensional spatial frequency transform of the facial image frames.

3. The method of claim 2, wherein the physical degradation signature further comprises an 8×8 block boundary gradient discontinuity metric measuring block-based video compression artifacts.

4. The method of claim 1, wherein the physical degradation signature comprises an inter-frame temporal motion variance metric measuring temporal smearing across consecutive frames of the input video.

5. The method of claim 1, wherein the physical degradation signature comprises a spatial Total Variation (TV) gradient energy metric and a luminance dynamic range clipping metric.

6. The method of claim 1, wherein the plurality of forensic expert branches comprises:
   - a facial-component-guided semantic branch modeling anatomical landmark consistency;
   - a spatiotemporal thumbnail layout branch modeling multi-frame coherence; and
   - a frequency-domain branch modeling synthetic generative artifacts via discrete cosine transform (DCT) frequency filtering.

7. The method of claim 6, wherein the facial-component-guided semantic branch comprises a frozen Vision Transformer backbone and learned spatial component queries attending to lips, eyes, and nose.

8. The method of claim 6, wherein the spatiotemporal thumbnail layout branch comprises a Swin Transformer processing multiple frames tiled into a two-dimensional thumbnail grid.

9. The method of claim 1, wherein the dynamic neural router comprises a multi-layer perceptron including layer normalization, non-linear activation, and a temperature-scaled Softmax output layer generating weights constrained to a probability simplex.

10. The method of claim 1, wherein the dynamic neural router is trained using a composite loss function comprising a binary cross-entropy classification loss and a Shannon entropy diversity loss that penalizes expert collapse across training batches.

11. A video inspection system for deepfake detection, comprising:
    - one or more processors; and
    - a memory storing executable instructions that, when executed by the one or more processors, cause the system to:
      - receive a digital video stream;
      - extract a physical degradation feature vector characterizing transmission channel distortions of the video stream;
      - process the video stream through a plurality of forensic expert neural networks comprising at least an anatomical component guidance network, a spatiotemporal thumbnail network, and a frequency-domain decomposition network;
      - dynamically predict instance-level routing coefficients for the forensic expert networks using a gating neural network taking the physical degradation feature vector as input;
      - normalize output margins of the forensic expert networks; and
      - generate an authenticity classification by calculating a weighted sum of the normalized output margins according to the predicted routing coefficients.

12. The system of claim 11, wherein the physical degradation feature vector comprises:
    - a two-dimensional Fourier power spectral roll-off ratio;
    - a horizontal and vertical block boundary discontinuity jump ratio; and
    - an inter-frame mean absolute difference variance.

13. The system of claim 11, wherein when the input video exhibits high-frequency attenuation characteristic of H.264 compression, the gating neural network automatically increases routing coefficients assigned to the frequency-domain decomposition network and the spatiotemporal thumbnail network relative to the anatomical component guidance network.

14. The system of claim 11, wherein the forensic expert networks are executed in parallel on a hardware neural network accelerator.

15. The system of claim 11, wherein the normalization of output margins comprises subtracting a canonical validation mean and dividing by a canonical validation standard deviation for each respective expert network.

16. The system of claim 11, wherein the system is deployed as an inline filter within a video conferencing gateway, social media upload pipeline, or cloud media streaming server.

17. The system of claim 11, wherein the instructions further cause the system to output an interpretable routing attribution visual displaying the relative contribution of each forensic expert network to the final authenticity classification.

18. The system of claim 11, wherein the gating neural network is trained using source-identity cluster cross-validation preventing identity-confounded shortcut learning.

19. A non-transitory computer-readable storage medium comprising instructions that, when executed by one or more processors, cause the one or more processors to perform operations comprising:
    - obtaining a sequence of facial video frames;
    - computing a non-semantic degradation signature comprising a 2D spectral decay slope, an 8×8 blockiness discontinuity score, and an inter-frame motion entropy score;
    - evaluating the video frames across a semantic foundation expert branch, a spatiotemporal thumbnail expert branch, and a frequency DCT expert branch to produce respective forensic margins;
    - generating dynamic expert weights via a multi-layer neural router conditioned on the degradation signature;
    - standardizing each forensic margin; and
    - combining the standardized forensic margins using the dynamic expert weights to output an authenticity score.

20. The non-transitory computer-readable storage medium of claim 19, wherein the neural router comprises a diversity-regularized objective maximizing Shannon entropy of batch-averaged routing weights to guarantee multi-expert utilization under varying transmission channel bandwidths.
