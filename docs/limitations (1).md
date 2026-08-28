# Limitations

Every project like this has weak spots. I'd rather list mine here than have someone else find them first.

## The validation leakage I found and fixed

This is the one I want to lead with, because it's the most significant thing that changed since the first version of this document. An earlier version of this pipeline used the Testing/ folder as the validation set during training, which meant hyperparameter selection, learning-rate scheduling, and early stopping were all guided by performance on the same 1,600 images later reported as the test set. That's leakage, and it's a more serious problem than the absence of external validation alone, because it means the originally reported accuracy wasn't measured on a genuinely held-out set.

I corrected this by carving out a proper 90/10 train/validation split from the Training/ folder only, and touching the Testing/ folder exactly once, for final evaluation. Re-running the full pipeline under this corrected protocol, accuracy moved from 91.44% to 91.31%, a small, honest confirmation that the original number wasn't wildly inflated, but the earlier protocol was still wrong on principle and needed fixing regardless of how much the number moved. Every figure in this document from here on reflects the corrected, leakage-free protocol.

## Not the most accurate model tested

Once I re-ran the model comparison cleanly, EfficientNetB3 (91.31%) was not the top performer: InceptionV3 came in at 92.19%, and MobileNetV2 at 91.75% using roughly a quarter of the parameters. I'm reporting this transparently rather than quietly keeping the old comparison table, since it directly affects how the backbone choice should be read. I'm keeping EfficientNetB3 for this pipeline because the spread across all four re-evaluated models is modest (about 1 percentage point) and because the point of this project is the uncertainty/explainability/fairness tooling around the backbone, not a claim of best-in-class accuracy. Reproducing the same pipeline on InceptionV3 or MobileNetV2 is a natural next step.

## Just one dataset

Everything in this project, training and evaluation, came from a single public source: the Masoud Nickparvar Brain Tumor MRI Dataset on Kaggle, 7,200 images split across four classes. One source means one preprocessing pipeline baked in before I ever touched it. I genuinely don't know how this model would hold up on scans from a different manufacturer, a different field strength (1.5T vs 3T), a different slice orientation, or a different hospital altogether, and honestly, I'd guess worse than what's reported here.

The real minimum bar for calling something clinically relevant is external validation on at least two independent datasets from different sites. I haven't done that, and I'm not going to pretend this project clears that bar.

## No clinical testing, period

This has never touched real, prospectively collected patient data inside an actual clinical workflow. Yes, the test-set numbers now come from a genuinely held-out split, but that's still retrospective performance, it doesn't tell you how the model behaves once image quality, patient mix, and random incidental findings start varying in ways a curated Kaggle dataset simply can't capture.

Getting this anywhere near a real deployment would mean IRB approval, a prospective pilot design, a radiologist watching over it the whole time, and tracking outcomes long after the fact.

## The demographic data isn't real

The fairness audit in Phase 4 relied on simulated age and gender labels, assigned based on published epidemiological patterns rather than anything collected from actual patients. There's no real demographic data attached to this dataset at all, which, to be fair, is normal for public medical imaging data given privacy rules.

So the numbers I got, a 2.76% gap by gender, 4.85% by age group, and an 11.8-percentage-point gap between the 61+ age group's share of high-uncertainty predictions and its share of the test set — show that the audit methodology works, not that these disparities exist in the real world. A genuine fairness audit needs verified demographic metadata, and getting that would mean IRB-approved data collection from real patients. I want to be upfront about this rather than let simulated results pass as something they're not.

## SHAP didn't work out

I originally planned to use SHAP DeepExplainer as my third XAI method. It turned out to be incompatible with TensorFlow 2.20/Keras 3 because of a deprecated `learning_phase` API, so I swapped in Integrated Gradients instead. I flagged this early, back in Phase 3 — the swap holds up fine on theoretical grounds, but it does mean my results aren't directly comparable to SHAP numbers reported elsewhere in the literature.

## Grad-CAM++ isn't as stable as it looks

Grad-CAM++ has a known fragility problem: two nearly identical images can produce saliency maps that look pretty different from each other. I didn't test for that stability systematically in this project. What I did notice is that the pituitary class had weaker anatomical localization than the other three, despite near-perfect classification accuracy, which makes me think the model might be leaning on features that aren't strictly anatomical for that class. Proper faithfulness metrics (deletion/insertion scores, SmoothGrad stability) would be needed to actually confirm that suspicion.

## Heavy blur breaks it, and now I've confirmed it with real motion blur too

When I ran the artifact robustness tests, severe Gaussian blur (kernel size 15) dropped accuracy from 91.31% down to about 55%. I originally flagged real motion-artifact simulation as untested future work, but I've since added it: directional motion blur at 5–15px, simulating patient head movement, dropped accuracy to as low as 57% at the longest length, consistent with the Gaussian-blur proxy, not just an artifact of it. I also added additive Gaussian sensor noise (distinct from Gaussian blur), which degraded performance more gradually, down to about 73% at the highest tested noise level. Real MRI scans with motion artifacts, poor field homogeneity, or acquisition noise could still easily produce unreliable predictions beyond what I've tested here (I haven't touched contrast variation from field inhomogeneity or elastic deformation). MC Dropout's uncertainty signal helps somewhat, uncertain predictions do get flagged, but I never specifically checked how well that uncertainty calibration holds up on degraded images.

## Built on Colab's free tier

The whole pipeline came together on Google Colab's free T4 GPU with 12GB of RAM, and that came with real headaches: no `.cache()` on datasets (it kept crashing RAM), some steps had to run on CPU, GPU quota ran out entirely partway through re-training the comparison models, and session disconnects meant I had to be careful and disciplined about checkpointing. It's reproducible within those limits, but a stronger compute setup would've let me run bigger ablations, more MC Dropout passes, a k-fold cross-validation pass (which I started but couldn't finish before quota ran out), and an actual hyperparameter search.

## Meningioma is still the tough one

It shows up as the weakest class no matter how I slice it: 85.0% accuracy, the highest ECE at 0.0616, the highest mean uncertainty (0.0457), and the largest gender-based equalized-odds gap of any class. This tracks with what's in the literature, meningiomas are visually inconsistent and can look a lot like gliomas depending on the slice, but it's still a real gap in the current model. A dedicated meningioma classifier, or a higher-resolution dataset with radiologist annotations, might be what closes it.

## What I'm claiming, and what I'm not

This project shows a way of building uncertainty-aware, explainable, fairness-audited medical image classifiers, evaluated under a validation protocol that I found broken and then fixed. It is not a clinically deployable system, and it is not a claim that EfficientNetB3 is the most accurate architecture for this task. I don't want either of those lines to get blurred.
