# Ethical Considerations

## Clinical Disclaimer

This is a research prototype, built for academic purposes. It has not been validated for clinical use, and it must not be used to inform, replace, or influence any medical diagnosis or treatment decision. Every prediction it makes should only ever be interpreted by qualified medical professionals who have full patient history, imaging context, and their own clinical judgment to work with. It doesn't meet the regulatory bar for medical device classification in any jurisdiction, and it was never built to.

## Fairness Audit and Demographic Limitations

The dataset behind this project, the Masoud Nickparvar Brain Tumor MRI Dataset on Kaggle, comes with no demographic metadata at all. No age, no gender, no ethnicity, no clinical history tied to any image. That's normal for public medical imaging datasets, given privacy regulations like HIPAA and GDPR, but it does mean I had to get creative to demonstrate a fairness audit at all.

What I did was assign simulated demographic labels to the test set, based on patterns documented in the neurological literature, glioma skews slightly male, meningioma is more common in older female patients, pituitary adenomas tend to show up in younger populations. I want to be explicit: these labels are simulated. They are not real patient data.

So when the audit turned up a 1.77% gender accuracy gap and a 3.59% age group gap, what that demonstrates is the methodology working as intended, not a real-world clinical disparity. Nobody should read these numbers as evidence of bias in an actual deployed system. Getting a real fairness audit would mean a dataset with verified demographic metadata and prospective clinical validation, neither of which I have here.

## Uncertainty Quantification and Human-in-the-Loop Design

The MC Dropout system was built to support a radiologist's judgment, not stand in for it. The rejection curve makes the case for this directly: defer the model's 20% most uncertain predictions to a human reviewer, and accuracy on what's left climbs from 91.4% to 98.1%. That handoff, the model knowing when to step back, is really the core design idea behind this whole project.

One finding I think is genuinely worth highlighting: the model's high-uncertainty predictions are disproportionately concentrated in the 61+ age group (36.9% of uncertain predictions, versus 26.3% of the overall test set) and in female patients (59.1% versus 53.2%). The clinical implication is fairly direct, the model is least confident on exactly the groups that tend to be underrepresented in typical training data. Uncertainty and demographic fairness don't usually get studied together in the medical imaging literature, and I think that connection is one of the more meaningful contributions of this project.

The deployment interface implements an uncertainty flag (std > 0.15 triggers a high-uncertainty warning) specifically so a clinician is never shown a confident-looking prediction on a case the model is actually unsure about.

## Model Limitations

**One dataset, one source.** Training and evaluation both happened on a single public dataset. How the model would perform on scans from different scanners, protocols, field strengths, or clinical sites is genuinely unknown, and probably worse than what's reported here. External validation on independent datasets would be necessary before this goes anywhere near deployment.

**No sense of time.** The model classifies static slices. It has no concept of tumor progression, treatment history, or follow-up imaging, and can't tell you whether a tumor is growing, shrinking, or post-surgical.

**Heavy blur is a real problem.** Robustness testing showed severe Gaussian blur (kernel=15) dropping accuracy from 91.4% to 56.8%. Motion artifacts, poor field homogeneity, or acquisition noise in real scans could easily produce unreliable predictions.

**Demographics are simulated, not real.** As noted above, the fairness audit ran on simulated metadata. Actual real-world disparities could look quite different.

**Never tested prospectively.** This model has never seen prospectively collected patient data in an actual clinical setting. Retrospective test-set numbers don't guarantee the same performance shows up in a live diagnostic workflow.

## Explainability Limitations

Grad-CAM++ heatmaps show which regions influenced a prediction, they are not medical explanations. I used three XAI methods (Grad-CAM++, LIME, Integrated Gradients) precisely because each one captures a different slice of model behavior, and because their disagreement is itself informative. Worth noting: the pituitary class had the weakest Grad-CAM++ alignment with the expected anatomical region of the four classes, a limitation I'm flagging here rather than glossing over.

## Data Privacy

No patient data was collected, stored, or processed anywhere in this project. The dataset is publicly available under a Creative Commons license on Kaggle and contains no identifiable patient information.

## What Responsible Deployment Would Actually Require

If this ever moved toward real clinical or research use, at minimum it would need:
- External validation on at least two independent datasets from different clinical sites
- A prospective pilot study with radiologist oversight and outcome tracking
- Real demographic metadata, collected properly, with a fairness re-audit on actual patient populations
- Regulatory review, FDA 510(k) or CE marking, depending on where it's deployed
- IRB approval for any patient-facing use
- Ongoing monitoring for performance drift and demographic disparities once it's actually in use
