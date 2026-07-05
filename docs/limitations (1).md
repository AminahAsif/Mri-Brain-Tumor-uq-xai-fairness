# Limitations

Every project like this has weak spots. I'd rather list mine here than have someone else find them first.

## Just one dataset

Everything in this project, training and evaluation, came from a single public source: the Masoud Nickparvar Brain Tumor MRI Dataset on Kaggle, 7,023 images split across four classes. One source means one preprocessing pipeline baked in before I ever touched it. I genuinely don't know how this model would hold up on scans from a different manufacturer, a different field strength (1.5T vs 3T), a different slice orientation, or a different hospital altogether, and honestly, I'd guess worse than what's reported here.

The real minimum bar for calling something clinically relevant is external validation on at least two independent datasets from different sites. I haven't done that, and I'm not going to pretend this project clears that bar.

## No clinical testing, period

This has never touched real, prospectively collected patient data inside an actual clinical workflow. Yes, the test-set numbers came from a properly held-out split, but that's still retrospective performance, it doesn't tell you how the model behaves once image quality, patient mix, and random incidental findings start varying in ways a curated Kaggle dataset simply can't capture.

Getting this anywhere near a real deployment would mean IRB approval, a prospective pilot design, a radiologist watching over it the whole time, and tracking outcomes long after the fact.

## The demographic data isn't real

The fairness audit in Phase 4 relied on simulated age and gender labels, assigned based on published epidemiological patterns rather than anything collected from actual patients. There's no real demographic data attached to this dataset at all, which, to be fair, is normal for public medical imaging data given privacy rules.

So the numbers I got, a 1.77% gap by gender, 3.59% by age group, show that the audit methodology works, not that these disparities exist in the real world. A genuine fairness audit needs verified demographic metadata, and getting that would mean IRB-approved data collection from real patients. I want to be upfront about this rather than let simulated results pass as something they're not.

## SHAP didn't work out

I originally planned to use SHAP DeepExplainer as my third XAI method. It turned out to be incompatible with TensorFlow 2.20/Keras 3 because of a deprecated `learning_phase` API, so I swapped in Integrated Gradients instead. I flagged this early, back in Phase 3, the swap holds up fine on theoretical grounds, but it does mean my results aren't directly comparable to SHAP numbers reported elsewhere in the literature.

## Grad-CAM++ isn't as stable as it looks

Grad-CAM++ has a known fragility problem: two nearly identical images can produce saliency maps that look pretty different from each other. I didn't test for that stability systematically in this project. What I did notice is that the pituitary class had weaker anatomical localization than the other three, despite near-perfect classification accuracy, which makes me think the model might be leaning on features that aren't strictly anatomical for that class. Proper faithfulness metrics (deletion/insertion scores, SmoothGrad stability) would be needed to actually confirm that suspicion.

## Heavy blur breaks it

When I ran the artifact robustness tests, severe Gaussian blur (kernel size 15) dropped accuracy from 91.4% all the way down to 56.8%. Real MRI scans with motion artifacts, poor field homogeneity, or acquisition noise could easily produce unreliable predictions without the model giving any visible warning. MC Dropout's uncertainty signal helps somewhat here, uncertain predictions do get flagged, but I never specifically checked how well that uncertainty calibration holds up on degraded images.

## Built on Colab's free tier

The whole pipeline came together on Google Colab's free T4 GPU with 12GB of RAM, and that came with real headaches, no `.cache()` on datasets (it kept crashing RAM), some steps had to run on CPU, and session disconnects meant I had to be careful and disciplined about checkpointing. It's reproducible within those limits, but a stronger compute setup would've let me run bigger ablations, more MC Dropout passes, and an actual hyperparameter search.

## Meningioma is still the tough one

It shows up as the weakest class no matter how I slice it, 86.5% accuracy, the highest ECE at 0.051, the highest uncertainty (mean std of 0.0476), and the largest equalized odds gap of any class. This tracks with what's in the literature, meningiomas are visually inconsistent and can look a lot like gliomas depending on the slice, but it's still a real gap in the current model. A dedicated meningioma classifier, or a higher-resolution dataset with radiologist annotations, might be what closes it.

## What I'm claiming, and what I'm not

This project shows a way of building uncertainty-aware, explainable, fairness-audited medical image classifiers. It is not a clinically deployable system, and I don't want that line to get blurred.
