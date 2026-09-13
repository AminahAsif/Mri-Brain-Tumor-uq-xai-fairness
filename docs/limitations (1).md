This has never touched real, prospectively collected patient data inside an actual clinical workflow. Yes, the test-set numbers now come from a genuinely held-out split, but that's still retrospective performance, it doesn't tell you how the model behaves once image quality, patient mix, and random incidental findings start varying in ways a curated Kaggle dataset simply can't capture. Getting this anywhere near a real deployment would mean IRB approval, a prospective pilot design, a radiologist watching over it the whole time, and tracking outcomes long after the fact.

## The demographic data isn't real

The fairness audit in Phase 4 relied on simulated age and gender labels, assigned based on published epidemiological patterns rather than anything collected from actual patients. There's no real demographic data attached to this dataset at all, which, to be fair, is normal for public medical imaging data given privacy rules. So the numbers I got, a 2.76% gap by gender, 4.85% by age group, and an 11.8-percentage-point gap between the 61+ age group's share of high-uncertainty predictions and its share of the test set — show that the audit methodology works, not that these disparities exist in the real world. A genuine fairness audit needs verified demographic metadata, and getting that would mean IRB-approved data collection from real patients. I want to be upfront about this rather than let simulated results pass as something they're not.

## SHAP didn't work out

I originally planned to use SHAP DeepExplainer as my third XAI method. It turned out to be incompatible with TensorFlow 2.20/Keras 3 because of a deprecated `learning_phase` API, so I swapped in Integrated Gradients instead. I flagged this early, back in Phase 3 — the swap holds up fine on theoretical grounds, but it does mean my results aren't directly comparable to SHAP numbers reported elsewhere in the literature.

## Grad-CAM++ isn't as stable as it looks

Grad-CAM++ has a known fragility problem: two nearly identical images can produce saliency maps that look pretty different from each other. I didn't test for that stability systematically in this project. What I did notice is that the pituitary class had weaker anatomical localization than the other three, despite near-perfect classification accuracy, which makes me think the model might be leaning on features that aren't strictly anatomical for that class. Proper faithfulness metrics (deletion/insertion scores, SmoothGrad stability) would be needed to actually confirm that suspicion.

## Heavy blur breaks it, and now I've confirmed it with real motion blur too

When I ran the artifact robustness tests, severe Gaussian blur (kernel size 15) dropped accuracy from 91.31% down to about 55%. I originally flagged real motion-artifact simulation as untested future work, but I've since added it: directional motion blur at 5–15px, simulating patient head movement, dropped accuracy to as low as 57% at the longest length, consistent with the Gaussian-blur proxy, not just an artifact of it. I also added additive Gaussian sensor noise (distinct from Gaussian blur), which degraded performance more gradually, down to about 73% at the highest tested noise level. Real MRI scans with motion artifacts, poor field homogeneity, or acquisition noise could still easily produce unreliable predictions beyond what I've tested here (I haven't touched contrast variation from field inhomogeneity or elastic deformation). MC Dropout's uncertainty signal helps somewhat, uncertain predictions do get flagged, but I never specifically checked how well that uncertainty calibration holds up on degraded images.

## Built on Colab's free tier

The whole pipeline came together on Google Colab's free T4 GPU with 12GB of RAM, and that came with real headaches: no `.cache()` on datasets (it kept crashing RAM), some steps had to run on CPU, GPU quota ran out entirely partway through re-training the comparison models, and session disconnects meant I had to be careful and disciplined about checkpointing. It's reproducible within those limits, but a stronger compute setup would've let me run bigger ablations, more MC Dropout passes, a k-fold cross-validation pass (which I started but couldn't finish before quota ran out), and an actual hyperparameter search.

## Meningioma is still the tough one

It shows up as the weakest class no matter how I slice it: 85.5% recall, the highest ECE at 0.0616, the highest mean uncertainty (0.0457), and the largest gender-based equalized-odds gap of any class. This tracks with what's in the literature, meningiomas are visually inconsistent and can look a lot like gliomas depending on the slice, but it's still a real gap in the current model. A dedicated meningioma classifier, or a higher-resolution dataset with radiologist annotations, might be what closes it.

## MobileNetV2 turned out to be a genuine trade-off, not a simple upgrade

I later ran the same full analysis, uncertainty, calibration, explainability, fairness, and robustness, on MobileNetV2, since it scored higher on raw accuracy (92.31% vs. 91.31%) and uses about a fifth of the parameters. It's not a clean win, though. Grad-CAM++ fails completely on this backbone; I checked the raw pre-normalization activation values directly, and they're near-zero across all four classes, so the heatmaps that min-max normalization produces are just noise dressed up to look like signal. MobileNetV2 is also noticeably more fragile under Gaussian blur and JPEG compression, accuracy drops to 34.6% and 64.7% respectively, versus 55.1% and about 85% for EfficientNetB3, though the two backbones are comparable under motion blur and sensor noise. That's why I stuck with EfficientNetB3 as the primary backbone rather than switching: the accuracy gain isn't free.

## What I'm claiming, and what I'm not

This project shows a way of building uncertainty-aware, explainable, fairness-audited medical image classifiers, evaluated under a validation protocol that I found broken and then fixed. It is not a clinically deployable system, and it is not a claim that EfficientNetB3 is the most accurate architecture for this task. I don't want either of those lines to get blurred.
