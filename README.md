MRI Brain Tumor Classification 
Uncertainty, Explainability, and Fairness

Amina Asif 


Most brain tumor MRI classifiers I came across while building this project report a single accuracy number and stop there. That bothered me , in a clinical setting, a model being "91% accurate" tells a radiologist almost nothing about whether to trust this particular prediction in front of them. So the question I built this project around is: can a classifier be made to communicate when it doesn't know, and can uncertainty quantification, explainability, and fairness auditing together produce something that looks more like a responsible decision-support tool than just another benchmark entry?

This is my answer to that question, built across five phases over several weeks of iterative work in Google Colab.


Live Demo

 Try it here → https://huggingface.co/spaces/AminahAsif/MRI-Brain-Tumor-Classification

Upload any brain MRI image and you get: a prediction, a confidence score, a Grad-CAM++ heatmap showing what the model looked at, and a color-coded uncertainty flag  generated from 50 Monte Carlo Dropout passes. If the model is uncertain, it says so , loudly.


Key Numbers

What Result EfficientNetB3 test accuracy 91.44% Scratch CNN baseline 88.94% McNemar test p-value0.00098 , improvement is statistically real Macro-average AUC (4-class)0.9863 Retain 80% most certain → accuracy 98.05% Incorrect predictions uncertainty 4.7× higher than correct onesGender accuracy gap (simulated audit) 1.77% Age group accuracy gap (simulated audit)3.59% 61+ age group in high-uncertainty pool36.9% vs 26.3% of total novel findingECE (calibration)0.0264 well-calibrated


The Rejection Curve 
Why It Matters

look at results folder

This is the result I'm most proud of. When the model defers its 20% most uncertain predictions to a human reviewer instead of forcing a guess, accuracy on the remaining cases jumps from 91.4% to 98.1%. Defer 45% and it hits 100%. The model genuinely knows when it doesn't know , and I can prove it statistically (Mann-Whitney p=1.68e-58 for correct vs incorrect uncertainty distributions).


What's in this repo

notebooks/
  MRI_1_setup_baseline_efficientnet.ipynb   ← baseline CNN, CLAHE, EfficientNetB3 training
  MRI_2_uncertainty_mcnemar.ipynb             ← MC Dropout, rejection curve, McNemar, ROC-AUC, ECE
  MRI_3_xai_explainability.ipynb              ← Grad-CAM++, LIME, Integrated Gradients
  MRI_4_fairness_audit.ipynb                  ← Fairlearn, AIF360, artifact robustness testing
  MRI_5_deployment.ipynb                      ← Gradio app, HuggingFace deployment

docs/
  literature_review.md          ← 13 papers across transfer learning, UQ, XAI, fairness, conformal prediction
  research_proposal.md          ← research question, identified gaps, clinical motivation
  xai_qualitative_analysis.md   ← per-class Grad-CAM++, LIME, IG findings and disagreements
  ethical_considerations.md     ← clinical disclaimer, bias disclosure, responsible deployment requirements
  limitations.md                ← honest limitations: single dataset, simulated demographics, Grad-CAM fragility


How I built this

Notebook 1 : Classification pipeline

I started by training a small scratch CNN (863K params) as a fixed baseline , 88.9% test accuracy, saved and locked. Then I preprocessed all 7,023 images with CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve tumor-tissue contrast, saving the results to Drive so I never have to rerun that step. EfficientNetB3 (ImageNet pretrained) was fine-tuned in two stages: frozen backbone first, then the last 30 layers unfrozen at LR=1e-5 with light augmentation. I used the Testing/ folder as validation during training so val accuracy equalled test accuracy , no optimistic gap.

The ablation study tells the story of what each component actually contributes:

VariantAccuracyvs FullNo CLAHE89.62%-1.81%Frozen backbone only87.38%-4.06%No augmentation90.38%-1.06%Full pipeline91.44%—

Notebook 2: Uncertainty quantification

I kept Dropout(0.3) active at inference time and ran 50 stochastic forward passes per image — but with a specific architectural choice: backbone features are extracted once with training=False (so BatchNorm uses learned population statistics), then only the dropout layer is stochastic across passes. This avoids the common mistake of calling model(x, training=True) which corrupts predictions by putting BatchNorm in training mode. The result is 50 valid probability distributions per image, from which I compute mean prediction and standard deviation as an uncertainty estimate.

I then ran four Mann-Whitney U tests to verify the uncertainty signal is statistically real , not just a curiosity. The most useful result: borderline glioma↔meningioma misclassifications have 2.4× higher uncertainty than correctly classified cases of the same classes (p<1e-20).

Notebook 3: Explainability

Three methods side by side , Grad-CAM++, LIME, and Integrated Gradients. SHAP was originally planned but is incompatible with TF 2.20/Keras 3, so I used Integrated Gradients as a theoretically equivalent replacement (it satisfies the same Shapley axioms without the library dependency). The most interesting finding was the pituitary class: 100% confidence, near-zero uncertainty, but the Grad-CAM++ heatmap doesn't align as cleanly with the sella turcica as the meningioma heatmap aligns with the dural attachment site. I flagged this as a limitation rather than hiding it.

Notebok 4: Fairness

No real demographic metadata exists in this dataset , that's standard for public medical imaging datasets due to patient privacy. I simulated age and gender labels based on epidemiological patterns in the literature and disclosed this clearly. The novel contribution here is connecting uncertainty and fairness: the 61+ age group accounts for 36.9% of high-uncertainty predictions but only 26.3% of the test set. The model is least confident on exactly the demographic group that's most underrepresented in typical training data. I haven't seen this analysis done in brain tumor MRI papers before.

AIF360 Reweighing brought gender disparate impact from 0.9808 to 1.0000. All equalized odds differences were below 0.1 (the standard acceptable threshold) for all four classes.

Notebook 5: Deployment

Gradio interface with an uncertainty flag (red if std > 0.15, green if std < 0.05, yellow in between), Grad-CAM++ overlay, and a clinical disclaimer at the top. Deployed on HuggingFace Spaces free tier.


Reproducing this project

Everything is saved to Google Drive — model checkpoints, predictions as numpy arrays, metrics as JSON files. Every notebook starts with a reload cell that restores all variables from Drive. SEED=42 is fixed throughout.

pythonSEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

Dataset: Masoud Nickparvar — Brain Tumor MRI Dataset


Supporting documents


Literature Review ->13 papers, why I chose each one, what I took from them
Research Proposal -> the question this project is trying to answer and why
XAI Qualitative Analysis —> honest per-class findings including the pituitary limitation
Ethical Considerations —> clinical disclaimer, what this can and cannot claim
Limitations —> single dataset, simulated demographics, Grad-CAM fragility, and more



 Clinical Disclaimer

This is a research prototype built for an academic portfolio. It is not validated for clinical use and must not be used to inform, replace, or influence any medical diagnosis or treatment decision. See docs.
