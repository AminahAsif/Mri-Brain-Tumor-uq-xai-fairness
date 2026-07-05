# Qualitative XAI Analysis, Grad-CAM++, LIME, and Integrated Gradients

## Why not just stick with one method

My first instinct, honestly, was to just go with Grad-CAM++ since it's the default choice in most medical imaging papers. That changed after I read a 2025 meta-analysis that scored Grad-CAM, LIME, and SHAP quite differently on fidelity, 0.54, 0.81, and 0.38 respectively, and showed their stability under noise shifts depending on the imaging modality. The takeaway for me was pretty clear: no single method wins across the board, and when two methods disagree, that disagreement is telling you something real about what the model is picking up on versus what each technique happens to be sensitive to.

So I ran all three methods on one correctly-classified test image per class, specifically the image closest to the median uncertainty for that class. I wasn't trying to re-validate the model's accuracy through the explanations; that job was already done statistically, via McNemar's test (p=0.00098) and macro ROC-AUC (0.9863). What I actually wanted to check was whether the model's attention lines up with anatomically sensible regions, and to be honest in the cases where it doesn't.

One thing worth flagging up front: SHAP DeepExplainer was supposed to be the third method here, but it turned out to be incompatible with TensorFlow 2.20/Keras 3 because of a deprecated `learning_phase` API. I swapped in Integrated Gradients instead, it produces Shapley-value-consistent attributions and satisfies the completeness and sensitivity axioms, so if anything it has a stronger theoretical footing than SHAP's DeepExplainer approximation for deep nets. I'd rather say that plainly than quietly swap it in and hope no one notices.

---

## Glioma (confidence: 99.7%, uncertainty: 0.0015)

Grad-CAM++'s main hotspot landed on the left temporal-parietal region of the axial slice, right where the tumor mass is visually obvious. LIME picked out several positive superpixels in the same area, with negative regions over the skull and background. Integrated Gradients gave a more diffuse, warm attribution map, with the strongest signal sitting near the tumor's boundary rather than its center.

All three agreed the tumor region was driving the classification. Where they diverged was spatial granularity, Grad-CAM++ stayed tightly localized while Integrated Gradients spread out further into the surrounding tissue. I don't actually see that as a problem. Gliomas are infiltrative and don't have clean margins, so it's entirely plausible the surrounding tissue does carry real discriminative signal, Integrated Gradients might just be more sensitive to picking that up than Grad-CAM++ is. Read that way, the disagreement is informative rather than a red flag.

---

## Meningioma (confidence: 92.2%, uncertainty: 0.0374)

This was, honestly, the strongest result of the four. Grad-CAM++ zeroed in tightly on the dural-based mass in the superior frontal region of the sagittal slice, exactly the spot where meningiomas typically arise, attached to the dura. LIME's red superpixels traced the tumor boundary with real precision. Integrated Gradients agreed, showing strong attribution over the same area.

The lower confidence (92.2%) and higher uncertainty (0.0374) here track with meningioma being the most diagnostically ambiguous class in this dataset, it's the one the confusion matrix shows getting mixed up most, with 54 meningioma cases predicted as pituitary and 37 as glioma. But the fact that all three XAI methods still land correctly on the dural attachment site, even while the model is expressing real uncertainty, tells me the model has actually learned something clinically meaningful here. It's just appropriately less confident about it than it is on the easier classes, which, if anything, is the behavior you'd want.

---

## Notumor (confidence: 99.8%, uncertainty: 0.0007)

Grad-CAM++ came back diffuse here, a broad heatmap across the upper cortex with no clear single hotspot. That's actually the result you'd want; there's no tumor to point at, and a model producing a sharp focal hotspot on a clean scan would be worrying, not comforting. LIME showed large green (positive) regions spread across normal brain tissue, with negative regions along the skull boundary, suggesting the model is basing its decision on what healthy brain architecture looks like rather than looking for something specific. Integrated Gradients showed a similarly spread-out pattern.

A diffuse, non-focal explanation is exactly what you'd hope for on a healthy scan. And it's worth noting that the MC Dropout uncertainty here, 0.0007, is the lowest of all four representative cases. The model isn't just getting it right; it knows it's getting it right.

---

## Pituitary (confidence: 100.0%, uncertainty: 0.0002)

Grad-CAM++ activated in the superior region of the sagittal slice. This is where I want to be upfront about a real limitation: pituitary adenomas arise from the sella turcica at the base of the brain, and while some of the activation did land near that expected location, the heatmap also spread into areas that aren't especially specific to pituitary pathology. LIME's positive superpixels covered a broad swath of the central brain, and Integrated Gradients followed a similar pattern.

Despite this being the model's highest-confidence, lowest-uncertainty prediction of the four, the spatial alignment here was actually the weakest. Accuracy on pituitary is 99.5%, and the model is 100% confident on this particular example, but the explanations suggest it might partly be relying on things that aren't strictly anatomical: overall brain geometry, contrast characteristics, scan orientation. High confidence without a clean anatomical explanation is a known failure mode in neural networks, and I think it's worth flagging even when the raw accuracy number looks great.

---

## Where the three methods agree, and where they don't

Across all four classes, Grad-CAM++ gave the most visually interpretable spatial localization, which fits its design as a class activation method. LIME's superpixel boundaries made positive and negative regions the easiest to read at a glance. Integrated Gradients was the most theoretically rigorous of the three, but also the subtlest, harder to interpret without laying it directly over the original scan.

All three methods agreed on the primary region of interest in every case. They also disagreed at finer spatial scales in every case. My honest read on that is that each method is measuring something slightly different, not that one of them is simply wrong. If I were a radiologist looking at this system, I'd treat agreement across all three as stronger evidence than any single method on its own.

---

## Limitations I want to flag upfront

Grad-CAM++ is fragile by design, two nearly identical images can sometimes produce noticeably different saliency maps. I came across this in a 2023 paper on black-box explainability in medical imaging, and it's a legitimate concern for anyone thinking of using this method clinically. I didn't test for that stability systematically in this project, and that's a real gap.

This whole analysis is based on four images, one per class. That's standard practice for qualitative XAI figures in published work, but it's not a statistically rigorous evaluation of explanation quality. A proper faithfulness check would need something like deletion/insertion scores, SmoothGrad stability analysis, or actual radiologist judgment across a much larger sample. I'd rather say that clearly than let four images sound like more evidence than they are.

And SHAP was dropped because of a framework incompatibility, not by choice, that's disclosed in the research proposal's limitations section too.
