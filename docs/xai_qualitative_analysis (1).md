# Qualitative XAI Analysis: Grad-CAM++, LIME, and Integrated Gradients

*Updated after the validation-leakage fix. Retraining the model under a corrected, leakage-free protocol changed which test-set image sits closest to the median uncertainty for each class, so the four representative examples analyzed below are not the same specific images as in the original version of this analysis. The confidence and uncertainty values are the actual outputs of the corrected model, not carried over from the earlier run.*

## Why not just stick with one method

My first instinct, honestly, was to just go with Grad-CAM++ since it's the default choice in most medical imaging papers. That changed after I read a 2025 meta-analysis that scored Grad-CAM, LIME, and SHAP quite differently on fidelity, 0.54, 0.81, and 0.38 respectively, and showed their stability under noise shifts depending on the imaging modality. The takeaway for me was pretty clear: no single method wins across the board, and when two methods disagree, that disagreement is telling you something real about what the model is picking up on versus what each technique happens to be sensitive to.

So I ran all three methods on one correctly-classified test image per class, specifically the image closest to the median uncertainty for that class. I wasn't trying to re-validate the model's accuracy through the explanations; that job was already done statistically, via McNemar's test (p=0.178 versus the scratch-CNN baseline, not statistically significant, under the corrected protocol) and macro ROC-AUC (0.9853). What I actually wanted to check was whether the model's attention lines up with anatomically sensible regions, and to be honest in the cases where it doesn't.

One thing worth flagging up front: SHAP DeepExplainer was supposed to be the third method here, but it turned out to be incompatible with TensorFlow 2.20/Keras 3 because of a deprecated `learning_phase` API. I swapped in Integrated Gradients instead, it produces Shapley-value-consistent attributions and satisfies the completeness and sensitivity axioms, so if anything it has a stronger theoretical footing than SHAP's DeepExplainer approximation for deep nets. I'd rather say that plainly than quietly swap it in and hope no one notices.

---

## Glioma (confidence: 99.5%, uncertainty: 0.0014)

Grad-CAM++'s hotspot landed centrally on the tumor mass in the axial slice, with the strongest activation over the region a radiologist would flag first. LIME's positive (green) superpixels covered a large share of the brain tissue around and including that region, with a smaller negative region elsewhere in the image. Integrated Gradients gave a more diffuse, speckled attribution pattern rather than one clean blob, with signal scattered across the tumor area and its margins.

All three point to the same general area as driving the classification. Where they diverge is spatial granularity, Grad-CAM++ stays more tightly localized while Integrated Gradients spreads its attribution more broadly. I don't see that as a problem. Gliomas are infiltrative and don't have clean margins, so it's entirely plausible the surrounding tissue does carry real discriminative signal that Integrated Gradients is more sensitive to picking up than Grad-CAM++ is. Read that way, the disagreement is informative rather than a red flag.

---

## Meningioma (confidence: 77.9%, uncertainty: 0.068)

This example is worth sitting with, because it looks quite different from the version of this analysis I wrote before the leakage fix. The confidence here (77.9%) is well below the 99%+ confidence seen on the other three representative examples, and the uncertainty (0.068) is the highest of the four by a wide margin.

Grad-CAM++ produced a focal hotspot on the sagittal slice in a plausible location for a meningioma (dural-based masses arise at the brain's surface, which is where the activation sits). LIME's superpixel map was more mixed than for the other classes: alongside the expected positive (green) region, it also picked out a negative (red) superpixel elsewhere in the image, which reads as the model finding some evidence against its own prediction even while ultimately landing on the right answer. Integrated Gradients gave a diffuse pattern without a single dominant hotspot.

This is not a coincidence relative to the rest of the pipeline: meningioma also carries the second-highest mean MC Dropout uncertainty of the four classes overall (0.0457, versus 0.0063 for no-tumor and 0.0070 for pituitary), and the lowest recall of the four classes in the per-class performance table. Taken together, this specific example is a fairly clean illustration of the model correctly flagging its own difficulty on a genuinely harder case, rather than being confidently wrong. I'd rather show this honest, lower-confidence example than cherry-pick a cleaner-looking meningioma case that wouldn't represent how the model actually behaves on this class.

---

## Notumor (confidence: 99.8%, uncertainty: 0.0006)

Grad-CAM++ came back diffuse here, without one sharp focal hotspot, that's the result you'd want, since there's no tumor to point at, and a model producing a tight hotspot on a clean scan would be worrying rather than reassuring. LIME showed a mix of positive (green) superpixels across a large area of normal brain tissue and a negative (red) region on one side of the image, consistent with the model weighing "does this look like healthy brain architecture" rather than searching for a specific lesion. Integrated Gradients gave a similarly non-focal, spread-out pattern.

A diffuse, non-focal explanation is exactly what you'd hope for on a healthy scan, and the MC Dropout uncertainty here (0.0006) is the lowest of all four representative cases. The model isn't just getting it right; it knows it's getting it right.

---

## Pituitary (confidence: 99.9%, uncertainty: 0.0008)

This is where I want to be upfront about a real limitation, and it held up even after the leakage fix. Pituitary adenomas arise from the sella turcica at the base of the brain, but the Grad-CAM++ activation for this example extended beyond that expected region into the lower face and neck area of the sagittal slice, territory that isn't anatomically specific to pituitary pathology. LIME's positive superpixels covered a broad area of the brain, and Integrated Gradients showed a similarly diffuse pattern without tightly tracking the sella turcica specifically.

Despite this being one of the model's highest-confidence, lowest-uncertainty predictions, the spatial alignment here was the weakest of the four classes. Per-class recall on pituitary is 98.75%, and the model is 99.9% confident on this particular example, but the explanations suggest it may partly be relying on cues that aren't strictly anatomical: overall image framing, contrast characteristics, or scan orientation that happen to correlate with pituitary cases in this dataset. High confidence without a clean anatomical explanation is a known failure mode in neural networks, and I think it's worth flagging even when, especially when the raw accuracy number looks great.

---

## Where the three methods agree, and where they don't

Across all four classes, Grad-CAM++ gave the most visually interpretable spatial localization, which fits its design as a class activation method. LIME's superpixel boundaries made positive and negative regions the easiest to read at a glance, and were also the method most likely to surface a dissenting (negative) region even on a correct prediction. Integrated Gradients was the most theoretically rigorous of the three, but also the subtlest, harder to interpret without laying it directly over the original scan.

All three methods pointed to a broadly consistent region of interest for glioma, no-tumor, and pituitary. For meningioma, the picture was murkier, consistent with that class's higher uncertainty. My honest read on this is that each method is measuring something slightly different, not that one of them is simply wrong. If I were a radiologist looking at this system, I'd treat agreement across all three as stronger evidence than any single method on its own, and I'd treat the meningioma-style disagreement as a signal to look closer, not to ignore.

---

## Limitations I want to flag upfront

Grad-CAM++ is fragile by design, two nearly identical images can sometimes produce noticeably different saliency maps. I came across this in a 2023 paper on black-box explainability in medical imaging, and it's a legitimate concern for anyone thinking of using this method clinically. I didn't test for that stability systematically in this project, and that's a real gap.

This whole analysis is based on four images, one per class. That's standard practice for qualitative XAI figures in published work, but it's not a statistically rigorous evaluation of explanation quality. A proper faithfulness check would need something like deletion/insertion scores, SmoothGrad stability analysis, or actual radiologist judgment across a much larger sample. I'd rather say that clearly than let four images sound like more evidence than they are.

The specific representative image chosen for each class is a function of which sample sits closest to the median uncertainty for that class in a given trained model, so, as this update itself demonstrates, retraining the model (even for a legitimate reason, like fixing a leakage bug) can change which case gets shown, and the qualitative texture of what it shows. That's worth keeping in mind when reading conclusions from any single representative-example figure like this one.

And SHAP was dropped because of a framework incompatibility, not by choice, that's disclosed in the research proposal's limitations section too.
