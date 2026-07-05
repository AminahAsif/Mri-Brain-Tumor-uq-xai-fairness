# Research Proposal: Uncertainty-Aware, Explainable, and Fairness-Audited Brain Tumor MRI Classification

## Research Question

Most brain tumor MRI classifiers I came across while researching this project report a single accuracy number and call it done. That bothered me, because in an actual clinical setting, hearing that a model is "94% accurate" tells a radiologist almost nothing about whether to trust the specific prediction sitting in front of them right now. So the question driving this whole project is: can a transfer-learning classifier be built to genuinely communicate when it doesn't know something, and can that self-reported uncertainty, paired with multiple explanation methods and a fairness audit, add up to something closer to a trustworthy clinical decision-support tool, rather than just one more accuracy benchmark?

I broke that down into three things I actually wanted to measure, not just assert:

First, if the model defers its most uncertain predictions to a radiologist instead of forcing a guess, does accuracy on what's left actually improve in a way I can prove statistically, not just eyeball off a nice-looking curve?

Second, when I run three different explainability methods (Grad-CAM++, LIME, and SHAP) on the same prediction, do they actually agree with each other, and with where the tumor visually sits? And when they don't agree, is that disagreement useful information, or just noise I need to explain away?

Third, and the part I haven't really seen done together anywhere in my reading, are the model's most uncertain predictions clustered in particular patient subgroups? Uncertainty quantification and fairness auditing tend to get treated as two completely separate problems in the papers I read. I wanted to find out if they're actually connected.

## Why this matters clinically

A missed glioma and a false alarm on a healthy scan both carry real consequences, and I don't think it's enough for a model to just be right most of the time, it needs to know when it's more likely to be wrong. That's really the whole motivation here. I'm not trying to build something that replaces a radiologist's judgment; I'm trying to show what a model would need to do before it could ever responsibly support one. A clinical disclaimer in the final demo makes that boundary explicit, this is a methodology demonstration, not a diagnostic tool, and I don't want that line blurred anywhere.

## What I found missing in the literature

I worked through 13 papers (the full list is in `/docs/literature_review.md`) covering transfer learning for MRI, Monte Carlo Dropout, XAI comparisons, and fairness auditing in healthcare AI. Three gaps kept showing up again and again.

The first: almost every brain tumor paper reports accuracy, F1, and AUC, and then stops, there's no mechanism in most of these pipelines for the model to say "I'm not sure about this one."

The second: papers using explainability tend to default to Grad-CAM alone, even though a 2025 meta-analysis I read scored Grad-CAM, LIME, and SHAP quite differently on fidelity (0.54, 0.81, and 0.38 respectively) and found their stability under noise varies substantially too. Very few brain tumor papers actually run all three side by side and grapple with what it means when they disagree.

The third gap is the one I find most interesting: none of the papers I reviewed asked whether a model's uncertain predictions disproportionately come from particular demographic groups. Fairness research and uncertainty research read like two separate conversations in the healthcare AI literature, even though once you've built both pieces, it's a pretty natural question to ask whether they're related.

My contribution here isn't a new method in any one of these three areas on its own, it's building a single pipeline where uncertainty, explainability, and fairness all sit together and actually get interrogated against each other.

## How I'm building this

I'm using EfficientNetB3, pretrained on ImageNet, fed CLAHE-preprocessed MRI input, fine-tuned in two stages: first a new head trained with the backbone frozen, then the last 30 layers unfrozen for a slower fine-tuning pass. I've already trained and locked in a from-scratch CNN baseline for comparison (88.9% test accuracy, 863K parameters), which lets me run a McNemar test later to prove statistically that any gain from EfficientNetB3 is real and not just noise.

For uncertainty, I'm using Monte Carlo Dropout, 50 stochastic forward passes per image at inference time, using the spread across those passes as the uncertainty signal. That feeds directly into a rejection curve: rank test images by uncertainty, abstain on the most uncertain ones, and see how much accuracy improves on what's left.

For explainability, every prediction runs through Grad-CAM++, LIME, and SHAP, laid out side by side against the original scan to see whether they agree with each other and with where the tumor actually is.

For fairness, I'm using Fairlearn and AIF360 to check equalized odds, demographic parity, and disparate impact across age and gender groups. I need to be upfront here: this dataset has no real demographic metadata attached to it, so I'll be using simulated demographic labels to demonstrate the audit methodology itself, not to claim I've found a real-world disparity. That gets stated clearly in the limitations section rather than left to look like an actual clinical finding.

Everything gets packaged into a Gradio app deployed on Hugging Face Spaces, showing the prediction, confidence, uncertainty score, and a Grad-CAM++ overlay, with a clinical disclaimer up top and a simple red/green warning tied to the model's uncertainty.

## Risks I'm keeping an eye on

Colab's free GPU sessions time out, so I'm saving every checkpoint to Drive right after training instead of risking a retrain from scratch. The ablation study, four model variants for the CLAHE/skull-strip comparison, is the piece most likely to burn through compute time, so I'm planning to spread it across a few sessions instead of trying to force it into one sitting. The missing demographic metadata is a limitation I knew about going in, not something I discovered halfway through, I'm handling it by being transparent about the simulation rather than dressing it up as a real finding. Hugging Face's free tier has its own resource limits, so I want to test the deployment early rather than leave it for the last week. Overall the roadmap gives me 5–8 weeks, and Phase 0 took about a week, so I'm treating the HIGH and STRETCH tasks as buffer space if a phase runs long, not something I'm locked into finishing no matter what.

## What I'm hoping to get out of this

Based on what similar EfficientNetB3 setups have hit in the literature, I'm targeting above 94% accuracy and an AUC above 0.97. On the uncertainty side, I want a rejection curve that shows a real, defensible accuracy improvement once the most uncertain 20% of predictions get set aside. For fairness, I'd rather honestly report one small, real disparity than claim a clean bill of health with no analysis backing it up. And by the end, I want an actual working public demo link, not just a notebook that only runs on my own machine.
