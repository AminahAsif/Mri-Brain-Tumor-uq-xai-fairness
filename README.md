
# MRI Brain Tumor Classification - Uncertainty, Explainability & Fairness

## Project Objective

This project builds a clinically-motivated brain tumor MRI classification pipeline that goes beyond accuracy reporting. By combining EfficientNetB3 transfer learning with Monte Carlo Dropout uncertainty quantification, three complementary XAI methods, and a fairness audit, the system is designed to communicate not just *what* it predicts, but *how confident* it is and *where* it looks ,the three things a radiologist would actually need before trusting a model's output.

The core finding: when the model defers its 20% most uncertain predictions to a human reviewer, accuracy on the remaining cases rises from 91.4% to 98.1%. Incorrect predictions carry 4.7× higher uncertainty than correct ones , proven statistically (Mann-Whitney p=1.68e-58). The model genuinely knows when it doesn't know.

---

## Live Demo

*(Click the image below to open the live demo on HuggingFace Spaces)*

<div align="center">
  <a href="https://huggingface.co/spaces/AminahAsif/MRI-Brain-Tumor-Classification">
    <img src="results/xai_4panel_final.png" alt="MRI Brain Tumor Classifier Demo" width="800" style="border-radius: 8px;">
  </a>
</div>

**[ Open Live Demo →](https://huggingface.co/spaces/AminahAsif/MRI-Brain-Tumor-Classification)**

Upload any brain MRI image and get: a tumor class prediction, confidence score, Grad-CAM++ heatmap, and a color-coded uncertainty flag (GREEN/ YELLOW /RED ) from 50 Monte Carlo Dropout passes.

---

## Key Results

| Metric | Result |
|--------|--------|
| EfficientNetB3 test accuracy | **91.44%** |
| Scratch CNN baseline | 88.94% |
| McNemar test (improvement significance) | **p = 0.00098** |
| Macro-average AUC (4-class OvR) | **0.9863** |
| Accuracy at 80% retention (rejection curve) | **98.05%** |
| Incorrect predictions uncertainty ratio | **4.7× higher** than correct |
| ECE (calibration error) | **0.0264**  well-calibrated |
| Gender accuracy gap (fairness audit) | 1.77% |
| Age group accuracy gap (fairness audit) | 3.59% |

---

## Rejection Curve

![Rejection Curve](results/rejection_curve.png)

*Accuracy climbs from 91.4% to 98.1% to 100% as the most uncertain predictions are deferred to a human reviewer.*

---

## System Architecture

The pipeline is structured across five phases, each building on the last.

### 1. Classification Backbone

- **Architecture:** EfficientNetB3 (ImageNet pretrained), fine-tuned in two stages , frozen head training first, then last-30-layer unfreeze at LR=1e-5
- **Preprocessing:** CLAHE applied once to all 7,023 images and saved to Drive , improves tumor-tissue contrast without processing overhead during training
- **Augmentation:** RandomFlip, RandomRotation(0.05), RandomZoom(0.1) , applied only during fine-tuning, not during frozen-head stage
- **Validation strategy:** Testing/ folder used as validation during training , val accuracy equals test accuracy, no optimistic gap
- **Ablation study:** Fine-tuning contributes most (-4.06% without it), CLAHE adds +1.81%, augmentation adds +1.06%

### 2. Uncertainty Quantification - Monte Carlo Dropout

- Dropout(0.3) kept active at inference  50 stochastic forward passes per image
- Backbone features extracted once with `training=False` so BatchNorm uses learned statistics  only the dropout layer is stochastic
- Per-image uncertainty = mean standard deviation across 50 prediction distributions
- Rejection curve validated: the model's uncertainty signal directly predicts its error likelihood

### 3. Explainability Panel - Three Methods

- **Grad-CAM++**  class activation heatmap on EfficientNetB3's `top_conv` layer
- **LIME**  superpixel perturbation (500 samples, 8 features)
- **Integrated Gradients**  path integral attribution, Shapley-value-consistent (used in place of SHAP which is incompatible with TF 2.20/Keras 3)

All three methods showed spatial agreement on tumor regions for glioma and meningioma. Disagreements at fine spatial scales are documented as informative rather than concealed.

### 4. Fairness Audit

- Simulated age and gender demographics based on epidemiological literature  disclosed clearly, not treated as real clinical data
- Fairlearn equalized odds difference < 0.1 for all four classes
- AIF360 Reweighing: gender disparate impact 0.9808 → 1.0000
- Novel finding: 61+ age group represents 36.9% of high-uncertainty predictions vs 26.3% of total  the model is least confident on the most underrepresented group
- Artifact robustness: JPEG compression robust (92% at q=50); severe blur degrades to 57%

### 5. Deployment

- Gradio 6.x interface with color-coded uncertainty flag (RED std > 0.15, GREEN std < 0.05)
- Grad-CAM++ overlay, confidence score, class probabilities table
- Clinical disclaimer at the top of the interface
- Deployed on HuggingFace Spaces (CPU Basic, free tier)

---

## XAI Panel

![XAI 4-Panel](results/xai_4panel_final.png)

*Rows: original MRI, Grad-CAM++, LIME superpixels, Integrated Gradients. Columns: glioma, meningioma, notumor, pituitary.*

---

## Quick Start

### Prerequisites

- Python 3.10+
- TensorFlow 2.20
- Google Colab (recommended) or local GPU

### Installation

```bash
pip install tensorflow opencv-python-headless numpy pandas scikit-learn \
            matplotlib seaborn scipy fairlearn aif360 gradio lime
```

### Run the demo locally

```bash
git clone https://github.com/AminahAsif/Mri-Brain-Tumor-uq-xai-fairness.git
cd Mri-Brain-Tumor-uq-xai-fairness
python deployment/app.py
```

### Reproduce the pipeline

Open any notebook in `notebooks/` in Google Colab. Each notebook starts with a reload cell that restores all variables from Drive checkpoints , no cell needs to be rerun from scratch after a session disconnect. All seeds are fixed at 42.

```python
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

Dataset: [Masoud Nickparvar — Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---


## Supporting Documents

- [Literature Review](docs/literature_review%20(1).md) - 13 papers, what I took from each one
- [Research Proposal](docs/research_proposal%20(2).md) - the question this project is answering and why it matters
- [XAI Qualitative Analysis](docs/xai_qualitative_analysis%20(1).md) - honest per-class findings including limitations
- [Ethical Considerations](docs/ethical_considerations%20(1).md) - clinical disclaimer and responsible deployment requirements
- [Limitations](docs/limitations%20(1).md) - what this project cannot claim and why
---

## Clinical Disclaimer

This system is a research prototype developed for academic purposes. It is **not validated for clinical use** and must **not** be used to inform, replace, or influence any medical diagnosis or treatment decision. See [docs/ethical_considerations.md](docs/ethical_considerations%20(1).md).

---


