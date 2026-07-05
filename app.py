import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE    = (224, 224)
UNC_HIGH    = 0.15
UNC_LOW     = 0.05

# Load model
print("Loading model.")
effnet_model  = tf.keras.models.load_model("model.keras")
backbone      = effnet_model.layers[1]
last_conv     = [l.name for l in reversed(backbone.layers)
                 if isinstance(l, tf.keras.layers.Conv2D)][0]
grad_model    = tf.keras.Model(
    inputs=backbone.input,
    outputs=[backbone.get_layer(last_conv).output, backbone.output]
)
dropout_layer = effnet_model.layers[2]
head_layer    = effnet_model.layers[3]
print(f"Model loaded. Last conv: {last_conv}")

DISCLAIMER = """
 **CLINICAL DISCLAIMER  READ BEFORE USE**
This system is a **research prototype only**, developed for academic demonstration.
It is **NOT validated for clinical use** and must **NOT** be used to inform, replace,
or influence any medical diagnosis or treatment decision.
All outputs must be interpreted by qualified medical professionals.
"""

def preprocess_image(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    img_pre = tf.keras.applications.efficientnet.preprocess_input(
                  img_resized.astype(np.float32))
    return img_resized, img_pre[np.newaxis]

def mc_dropout_predict(img_pre, n_passes=50):
    features = backbone(img_pre, training=False).numpy()
    feat_tf  = tf.constant(features)
    preds = []
    for _ in range(n_passes):
        drop = dropout_layer(feat_tf, training=True)
        pred = head_layer(drop, training=False).numpy()
        preds.append(pred[0])
    preds       = np.array(preds)
    mc_mean     = preds.mean(axis=0)
    mc_std      = preds.std(axis=0)
    uncertainty = float(mc_std.mean())
    confidence  = float(mc_mean.max())
    pred_class  = int(np.argmax(mc_mean))
    return mc_mean, uncertainty, confidence, pred_class

def make_gradcam(img_pre, pred_class, img_display):
    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            with tf.GradientTape() as t0:
                inp = tf.cast(img_pre, tf.float32)
                co, pred = grad_model(inp)
                t0.watch(co); t1.watch(co); t2.watch(co)
                loss = pred[:, pred_class]
            g1 = t0.gradient(loss, co)
        g2 = t1.gradient(g1, co)
    g3 = t2.gradient(g2, co)
    co=co[0]; g1=g1[0]; g2=g2[0]; g3=g3[0]
    s   = tf.reduce_sum(co, axis=(0,1))
    den = 2*g2 + s[None,None,:]*g3
    den = tf.where(den==0, tf.ones_like(den), den)
    w   = tf.reduce_sum((g2/den)*tf.nn.relu(g1), axis=(0,1))
    h   = tf.nn.relu(tf.reduce_sum(w*co, axis=-1)).numpy()
    h   = cv2.resize(h, IMG_SIZE)
    if h.max() > 0:
        h = (h-h.min())/(h.max()-h.min())
    colormap = matplotlib.colormaps["jet"]
    hc = (colormap(h)[:,:,:3]*255).astype(np.uint8)
    return cv2.addWeighted(img_display, 0.55, hc, 0.45, 0)

def predict_mri(image):
    if image is None:
        return None, "Please upload an MRI image.", ""
    img_display, img_pre = preprocess_image(image)
    mc_mean, unc, conf, pred_class = mc_dropout_predict(img_pre)
    overlay = make_gradcam(img_pre, pred_class, img_display)
    if unc > UNC_HIGH:
        flag = f" HIGH UNCERTAINTY (std={unc:.4f}) means Radiologist review REQUIRED"
    elif unc < UNC_LOW:
        flag = f" LOW UNCERTAINTY (std={unc:.4f}) means Model is confident"
    else:
        flag = f" MODERATE UNCERTAINTY (std={unc:.4f}) means Consider radiologist review"

    result = f"""## Prediction: {CLASS_NAMES[pred_class].upper()}
**Confidence:** {conf:.1%}
**MC Dropout Uncertainty:** {unc:.4f} (50 stochastic passes)
**Status:** {flag}


### Class Probabilities
| Class | Probability |

| Glioma | {mc_mean[0]:.4f} ({mc_mean[0]*100:.1f}%) |
| Meningioma | {mc_mean[1]:.4f} ({mc_mean[1]*100:.1f}%) |
| No Tumor | {mc_mean[2]:.4f} ({mc_mean[2]*100:.1f}%) |
| Pituitary | {mc_mean[3]:.4f} ({mc_mean[3]*100:.1f}%) |
"""
    return overlay, result, flag

with gr.Blocks(title="Brain Tumor MRI Classifier") as demo:
    gr.Markdown("#  Brain Tumor MRI Classification")
    gr.Markdown("### EfficientNetB3 + MC Dropout Uncertainty + Grad-CAM++ XAI")
    gr.Markdown(DISCLAIMER)
   
    gr.Markdown("**Model:** EfficientNetB3 | **Accuracy:** 91.44% | **AUC:** 0.9863 | **MC Dropout:** 50 passes")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload MRI Scan", height=300)
            predict_btn = gr.Button(" Analyze MRI", variant="primary", size="lg")
            gr.Markdown("**Classes:** Glioma | Meningioma | No Tumor | Pituitary")
        with gr.Column(scale=2):
            gradcam_output = gr.Image(label="Grad-CAM++ Explanation", height=300)
            uncertainty_flag = gr.Markdown("Upload an image and click Analyze.")
            result_output = gr.Markdown()
    gr.Markdown("---")
    gr.Markdown("""
### Uncertainty thresholds
-  **Green** (std < 0.05): Model is confident
-  **Yellow** (0.05 ≤ std ≤ 0.15): Moderate means consider expert review
-  **Red** (std > 0.15): High means radiologist review REQUIRED

###  This is NOT a clinical tool. For research demonstration only.
                                                                 Developed by Amina Asif 
""")
    predict_btn.click(
        fn=predict_mri,
        inputs=[image_input],
        outputs=[gradcam_output, result_output, uncertainty_flag]
    )

demo.launch()
