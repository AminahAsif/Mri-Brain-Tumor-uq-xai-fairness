# Literature Review 
 MRI Brain Tumor Classification with Uncertainty Quantification, Explainability, and Fairness

**Amina Asif**

## Why these papers

Before committing to an architecture and methodology for this project, I wanted to ground every major design decision in existing literature rather than picking methods arbitrarily. I organized my reading around the five components my pipeline needed to justify: which backbone to fine-tune, how to quantify uncertainty in a clinically meaningful way, which explainability tools actually hold up under scrutiny, how fairness auditing is done in healthcare AI, and whether conformal prediction is a viable complement to Monte Carlo Dropout. Thirteen papers ended up being enough to answer all five questions with reasonable confidence.

## 1. Why EfficientNetB3

I started here because the original scratch CNN baseline (which I reproduced in Phase 0) was never going to compete with a pretrained backbone, and I needed evidence for which pretrained backbone was actually worth the compute.

Islam et al.'s 2024 study in the *International Journal of Intelligent Systems* (Wiley) systematically swept the entire EfficientNet family  B0 through B7  on a 3,064-image T1-weighted contrast-enhanced MRI dataset, and EfficientNetB3 came out as the standout performer among the family. That was the single most direct piece of evidence pointing me toward B3 specifically rather than a smaller or larger variant.

A 2023 *Scientific Reports* paper independently fine-tuned EfficientNet variants on the CE-MRI dataset and benchmarked them against prior state-of-the-art methods from 2019–2023, reporting accuracy, precision, recall, and specificity alongside parameter count and FLOPs  which is where I got confirmation that the accuracy gains from larger EfficientNet variants stop being worth the computational cost past a certain point, reinforcing B3 as a sensible mid-size choice rather than going straight to B7.

A 2024 MDPI paper compared five other pretrained backbones (ResNet50, Xception, EfficientNetV2-S, ResNet152V2, VGG16) on the exact same 4-class glioma/meningioma/pituitary/notumor split I'm using, which gave me a ready-made reference point for the model comparison table I'll build in Phase 2.

Finally, since my dataset has more glioma and meningioma cases than other classes, I looked at an arXiv preprint from 2023 that evaluated 8 transfer-learning backbones combined with extra CNN layers specifically to handle class imbalance this is what shaped my decision to compute class weights via `tf.keras.utils.class_weight` rather than just relying on data augmentation alone.

## 2. Why Monte Carlo Dropout for uncertainty quantification

The core argument of my project is that a model should be able to say "I'm not sure"  and MC Dropout is the most practical way to get that signal without redesigning the network as a full Bayesian neural net.

Avci et al. (arXiv:2112.01587) is the paper I leaned on most heavily for the actual mechanics: keeping dropout active during inference (not just training) and averaging multiple stochastic forward passes, which both improves predictive accuracy and produces a usable confidence/uncertainty estimate per prediction. This is essentially the blueprint for my Phase 2 implementation  Dropout(0.3) before the final dense layer, `training=True` at inference, N=50 forward passes, mean and standard deviation computed per test image.

A 2023 review (PMC) on uncertainty quantification frameworks in medical imaging helped me understand the distinction between aleatoric uncertainty (irreducible noise in the data itself) and epistemic uncertainty (what the model doesn't know due to limited training data)  and made the case that giving clinicians a probabilistic basis for trusting or rejecting a model's output is exactly the kind of human-AI collaboration framing I want my rejection curve to demonstrate.

## 3. Why I'm using three XAI methods instead of one

My first instinct was to just use Grad-CAM since it's the most common in medical imaging papers, but a 2025 meta-analysis (PMC) covering 67 studies across radiology, pathology, and ophthalmology changed my approach. It quantified fidelity scores for each method  LIME at 0.81, Grad-CAM at 0.54, SHAP at 0.38  and showed that stability under noise varies a lot by modality. No single method came out reliably best across the board, which is the actual justification for why my 4-panel figure (original MRI, Grad-CAM++, LIME, SHAP) uses all three rather than picking a "winner": they're capturing different things and disagreement between them is itself informative.

A 2024 arXiv paper that applied Grad-CAM, LIME, and SHAP specifically to brain tumor detection (not just generic medical imaging) was the closest existing precedent I found to what I'm building, and it shaped how I plan to lay out the comparison  heatmap, superpixel regions, and pixel-level attribution side by side.

I also want to flag a known weakness going in rather than discover it later: a 2023 paper on black-box explainability for medical classifiers pointed out that Grad-CAM is fragile — two nearly identical images can produce noticeably different saliency maps. I'm citing this upfront in my limitations section rather than treating Grad-CAM's output as ground truth.

## 4. Why fairness auditing matters here, and how I'm doing it

I almost skipped a fairness analysis since this dataset has no demographic metadata, but a 2024 PMC paper on mitigating algorithmic bias in cardiovascular imaging convinced me it's still worth doing properly, even with the limitation of simulated demographic data. That paper used Fairlearn metrics  demographic parity, equalized odds, disparate impact  on a cardiovascular imaging pipeline and showed fairness-aware mitigation measurably improving equity, which is essentially the template I'm following for my own Phase 4 audit with Fairlearn and AIF360.

A broader review of fairness toolkits in computational medicine gave me the vocabulary and citation backbone for the Ethical Considerations section I'll need to write  particularly around why relying on accuracy alone can hide systematic underperformance on smaller subgroups.

## 5. Conformal prediction as a stretch goal

This wasn't part of my original plan, but reading two recent papers from the 2024 CHI conference and a 2024 arXiv survey made me want to include it as an optional addition if time allows. The CHI paper tested RAPS (Regularized Adaptive Prediction Sets) on image classification and found it achieves close to its target coverage guarantee (~95% at α=0.05) even under distribution shift, and that people reported more trust in RAPS prediction sets than plain Top-1 predictions. Unlike MC Dropout, which is a Bayesian approximation, conformal prediction gives a distribution-free statistical guarantee — so if I have time after the core MC Dropout pipeline is done, adding RAPS as a fourth uncertainty method would let me compare a heuristic approach against one with formal coverage guarantees.

---

## References

1. Islam, M. et al. (2024). BrainNet: Precision Brain Tumor Classification with Optimized EfficientNet Architecture. *International Journal of Intelligent Systems*. https://onlinelibrary.wiley.com/doi/10.1155/2024/3583612

2. (2023). Detection and classification of brain tumor using hybrid deep learning models. *Scientific Reports*. https://www.nature.com/articles/s41598-023-50505-6

3. (2024). Transfer-Learning Approach for Enhanced Brain Tumor Classification in MRI Imaging. *MDPI*. https://www.mdpi.com/2673-7426/4/3/95

4. (2023). Optimizing Brain Tumor Classification: A Comprehensive Study on Transfer Learning and Imbalance Handling in Deep Learning Models. *arXiv:2308.06821*.

5. Avci, M.Y. et al. Improving accuracy and uncertainty quantification of deep learning based quantitative MRI using Monte Carlo dropout. *arXiv:2112.01587*.

6. (2023). A Deep Learning-Based Framework for Uncertainty Quantification in Medical Imaging Using the DropWeak Technique. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9955446

7. (2025). Beyond Post hoc Explanations: A Comprehensive Framework for Accountable AI in Medical Imaging. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC12383817

8. (2024). Explainability of Deep Neural Networks for Brain Tumor Detection. *arXiv:2410.07613*.

9. (2023). MRxaI: Black-Box Explainability for Image Classifiers in a Medical Setting. *arXiv:2311.14471*.

10. (2024). Mitigating Algorithmic Bias in AI-Driven Cardiovascular Imaging for Fairer Diagnostics. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC11640708

11. Algorithmic fairness in computational medicine. *PMC*. https://pmc.ncbi.nlm.nih.gov/articles/PMC9463525

12. (2024). Evaluating the Utility of Conformal Prediction Sets for AI-Advised Image Labeling. *Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems*. https://dl.acm.org/doi/full/10.1145/3613904.3642446

13. (2024). Conformal Prediction for Class-wise Coverage via Augmented Label Rank Calibration. *arXiv:2406.06818*.

