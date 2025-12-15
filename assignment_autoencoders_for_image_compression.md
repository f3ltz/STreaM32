# 🧠 Assignment: Lightweight Autoencoders for Image Compression (TensorFlow)

## 🎯 Objective

In this assignment, you will:

1. Implement a **lightweight convolutional autoencoder (AE)** for image compression
2. Train it using **MSE and/or** **MAE loss functions**
3. Apply **Post-Training Quantization (PTQ)** and evaluate its impact
4. Apply **Quantization-Aware Training (QAT)** and evaluate its impact
5. Compare **floating-point, PTQ, and QAT models** using **compression + system metrics**



---

## 🧰 Tools & Resources

> **Execution environment:** You are expected to implement this assignment as a **Jupyter Notebook** and run it on **Google Colab or Kaggle** due to compute requirements. Local execution is optional but not recommended.

### Required

- Python 3.8+
- TensorFlow ≥ 2.10
- TensorFlow Model Optimization Toolkit
- NumPy, Matplotlib
- scikit-image (for PSNR, SSIM)

### Dataset

- CIFAR-10



### Guides

- **Autoencoder guide (TensorFlow):**\
  [https://www.tensorflow.org/tutorials/generative/autoencoder](https://www.tensorflow.org/tutorials/generative/autoencoder)

- **Post-Training Quantization:**\
  [https://www.tensorflow.org/lite/performance/post\_training\_quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

- **Quantization-Aware Training:**\
  [https://www.tensorflow.org/model\_optimization/guide/quantization/training](https://www.tensorflow.org/model_optimization/guide/quantization/training)

---

## 🧱 Part 1 — Baseline Lightweight Autoencoder (FP32)

### 1.1 Model Architecture

Implement a **lightweight convolutional autoencoder**.

Constraints:

- Use **Conv2D + Stride** for downsampling
- Use **Conv2DTranspose or UpSampling + Conv** for upsampling
- Keep parameter count **low** (≤ \~200k parameters recommended)

Example (you may modify):

```python
Input (32×32×3)
↓ Conv (stride=2)
↓ Conv (stride=2)
→ Bottleneck (8×8×C)
↑ Deconv
↑ Deconv
Output (32×32×3)
```

📌 **Deliverables**:

- Model summary
- Bottleneck tensor shape

---

### 1.2 Training

Train **two versions** of the AE:

1. AE trained with **Mean Squared Error (MSE)**
2. AE trained with **Mean Absolute Error (MAE)**

```python
loss = tf.keras.losses.MeanSquaredError()
# or
loss = tf.keras.losses.MeanAbsoluteError()
```

Train until convergence (e.g., 30–50 epochs).

---

## 📦 Part 2 —  Metrics (FP32 Model)

For the trained FP32 model, compute the following:

### 2.1 Compression Ratio

```text
Compression Ratio = Original Image Size / Latent Representation Size
```

- Original image size: assume 8 bits per pixel
- Latent size: number of latent elements × bits per element

---

### 2.2 Quality Metrics

Compute for test images:

- **PSNR (Peak Signal-to-Noise Ratio)**\
  Measures reconstruction fidelity in dB (higher is better)

- **SSIM (Structural Similarity Index)**\
  Measures perceptual/structural similarity (range 0–1)

Use:

```python
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
```

---

### 2.3 Inference Speed

Measure:

- Average inference time per image (ms)
- Batch size = 1

Use `time.perf_counter()` and report **mean over ≥100 runs**.

---

### 2.4 Model Size

Report:

- SavedModel or `.h5` file size (MB)

---

## ⚙️ Part 3 — Post-Training Quantization (PTQ)

### 3.1 Apply PTQ

Convert the trained FP32 model to **INT8** using TensorFlow Lite:

- Quantize **weights + activations**
- Use a small calibration dataset

---

### 3.2 Evaluation (PTQ)

Repeat **ALL metrics** from Part 2:

- Compression ratio
- PSNR
- SSIM
- Inference speed
- Model size

📌 Compare PTQ results with FP32 and explain:

- Accuracy drop (if any)
- Speed improvement
- Size reduction

---

## 🧪 Part 4 — Quantization-Aware Training (QAT)

### 4.1 Apply QAT

- Insert fake-quantization ops using TF Model Optimization Toolkit
- Retrain the model (10–20 epochs recommended)

📌 Use the **same architecture** as baseline AE.

---

### 4.2 Evaluation (QAT)

Repeat **ALL metrics** again:

- Compression ratio
- PSNR
- SSIM
- Inference speed
- Model size

📌 Compare **FP32 vs PTQ vs QAT** and discuss:

- Why QAT recovers quality
- Trade-offs in training complexity

---

## 📊 Part 5 — Final Comparison Table

Create a table like:

| Model | Loss    | PSNR | SSIM | Compression Ratio | Inference Time (ms) | Model Size (MB) |
| ----- | ------- | ---- | ---- | ----------------- | ------------------- | --------------- |
| FP32  | MSE/MAE |      |      |                   |                     |                 |
| PTQ   | MSE/MAE |      |      |                   |                     |                 |
| QAT   | MSE/MAE |      |      |                   |                     |                 |

---

## 📁 Submission Requirements

- Python notebooks / scripts
- Final comparison table
- Clear comments and plots

###

