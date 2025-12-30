

# Assignment: Neural Video Compression for Edge Deployment (STM32N6)

## **1. Overview & Objectives**
**Goal:** Build a Neural Video Codec capable of compressing real-world video streams **frame-by-frame** (Streaming Mode).
**Target Hardware:** STM32N6 (Cortex-M55 + Ethos-U55 NPU).
**Key Constraint:** You cannot see future frames. You can only use the **Current Frame ($x_t$)** and a **Memory Buffer ($S_{t-1}$)** from the previous step.

**Learning Outcomes:**
1.  Understanding **Streaming Inference** with TensorFlow (Stateful Deep Learning).
2.  Designing architectures for **Quantization (Int8)**.
3.  Managing **SRAM Memory Constraints** manually.

---

## **2. The Rules of the Game**

### **Hardware & Model Limits**
* **Max Model Size (Unquantized):** **20 MB** (approx. 5 Million Parameters in Float32).
    * *Note: This is your storage budget on the external flash.*
* **Max SRAM State Buffer:** **320 KB** (This holds your "Previous Frame" or "Hidden State").
    * *Note: This is your fast RAM budget. If you exceed this, the NPU stalls.*
* **Operations:** Must be compatible with Int8 Quantization (Avoid `tf.exp`, `tf.sigmoid` in critical paths; use `ReLU6`).

### **Dataset: Real-World Video**

1.  **Recommended for Training:** [Vimeo-90K (Septuplet)](http://toflow.csail.mit.edu/)

**Preprocessing Requirement:**
* You must downsample videos to **$128 \times 128$** to fit the STM32 compute budget.

---

## **3. The "State Strategy" (TensorFlow Boilerplate)**
In standard Keras `model.fit()`, state is usually hidden. For deployment, we must manage the state tensor manually.

**Everyone must use this base structure.**

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class StreamableModel(Model):
    def __init__(self):
        super(StreamableModel, self).__init__()
        # Define layers here
        # e.g., self.encoder = layers.Conv2D(...)
        pass

    def get_initial_state(self, batch_size):
        """
        Returns a tensor of zeros representing the initial state S_0.
        Shape must match what your model expects.
        """
        # Example: return tf.zeros((batch_size, 64, 64, 1))
        raise NotImplementedError

    def call(self, inputs):
        """
        Standard call for Keras (optional, can just use call_step for clarity)
        """
        return self.call_step(inputs[0], inputs[1])

    def call_step(self, x_t, state_t_minus_1):
        """
        Single Step Inference:
        Inputs:
            x_t: Current Frame [Batch, 64, 64, 1]
            state_t_minus_1: The memory from the previous step.
        Returns:
            x_hat_t: Reconstructed Current Frame
            state_t: New memory to save for the next step.
            latent: The compressed representation.
        """
        raise NotImplementedError
```



### **Helper: The Custom Training Loop**

We cannot use standard `model.fit()` easily because we need to handle the state passing between frames manually.



```
@tf.function
def train_step_streaming(model, video_batch, optimizer, loss_fn):
    # video_batch shape: [Batch, Time, Height, Width, Channels]
    
    batch_size = tf.shape(video_batch)[0]
    time_steps = tf.shape(video_batch)[1]
    
    # 1. Initialize State (S_0)
    state = model.get_initial_state(batch_size)
    
    total_loss = 0.0
    
    with tf.GradientTape() as tape:
        # 2. Loop through time (Frame by Frame)
        for i in range(time_steps):
            current_frame = video_batch[:, i] 
            
            # Run Inference
            reconstruction, next_state, latent = model.call_step(current_frame, state)
            
            # Calculate Loss
            loss = loss_fn(current_frame, reconstruction)
            total_loss += loss
            
            # 3. Update State
            # In TF, we usually don't need to detach manually inside the tape 
            # unless we want Truncated BPTT. For now, we let gradients flow.
            state = next_state

    # Backprop
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    return total_loss / tf.cast(time_steps, tf.float32)

```

----------

## **4. The Three Tracks (Choose One)**

### **Track A: The "Residual" Engineer**

_Focus: It is cheaper to compress the difference between frames than the frames themselves._

-   **Architecture 1: The Subtractor**
    
    -   **Input:** Current Frame $x_t$.
        
    -   **State:** The previous _reconstructed_ frame $\hat{x}_{t-1}$.
        
    -   **Logic:**
        
        1.  Compute Residual: `res = x_t - state`
            
        2.  Encode `res` $\to$ Latent $\to$ Decode `res_hat`.
            
        3.  Output: `x_hat = state + res_hat`.
            
-   **Architecture 2: The Concatenator**
    
    -   **Input:** Concatenate $x_t$ and State $\hat{x}_{t-1}$ along the channel axis.
        
    -   **TF Hint:** `x_input = tf.concat([x_t, state], axis=-1)`
        
    -   **Logic:** A CNN that learns optical flow implicitly.
        

### **Track B: The "Codebook" Quantizer**

_Focus: Use a discrete lookup table (Vector Quantization) to force high compression._

-   **Architecture 1: Simple VQ-VAE**
    
    -   **State:** None (Stateless).
        
    -   **Logic:** Use a Vector Quantizer layer (custom Keras layer required).
        
    -   **Goal:** Establish a baseline. Check for "flickering" artifacts.
        
-   **Architecture 2: Momentum VQ-VAE**
    
    -   **State:** The previous latent indices.
        
    -   **Logic:** Add a custom regularization term to the loss that penalizes the codebook indices from switching if the pixel difference is small.
        

### **Track C: The "Recurrent" Architect**

_Focus: Use a hidden vector to "remember" motion without storing full images._

-   **Architecture 1: The Bottleneck GRU**
    
    -   **State:** A hidden vector $h$ (e.g., shape `[Batch, 64]`).
        
    -   **Logic:**
        
        1.  Encoder $\to$ `z`.
            
        2.  `GRUCell` update: `new_h, new_z = gru_cell(z, h)`.
            
        3.  Decoder `new_z` $\to$ Image.
            
-   **Architecture 2: The Feature Buffer**
    
    -   **State:** The feature map from the _first_ layer of the Encoder.
        
    -   **Logic:**
        
        1.  `f_t = conv1(x_t)`
            
        2.  `f_mixed = (f_t + state) / 2` (Temporal smoothing).
            
        3.  Pass `f_mixed` to the rest of the network.
            

----------

## **5. Submission Requirements & Metrics**

You must submit a table comparing your model against a standard **H.264** baseline.

### **A. Neural Metrics**

Report these for your best model on the Test Set:

1.  **PSNR (Peak Signal-to-Noise Ratio):**
    
    -   Measure of reconstruction quality.
        
    -   _Formula:_ `tf.image.psnr(original, reconstructed, max_val=1.0)`
        
    
        
2.  **SSIM (Structural Similarity Index):**
    
    -   Measure of perceived visual quality.
        
    -   _Formula:_ `tf.image.ssim(original, reconstructed, max_val=1.0)`
        
    
        
3.  **Bits Per Pixel (BPP):**
    
    -   Calculated from your latent space size.
        
    -   _Formula:_ `(Latent_Size_Bits) / (Height * Width)`
        

### **B. The "H.264 Shootout" (Comparison)**

You must verify if your AI model is actually better than standard compression at the same file size.

**Steps to Benchmark:**

1.  **Calculate your model's bitrate:**
    
    -   Assume 30 FPS.
        
    -   `Bitrate (kbps) = BPP * Height * Width * 30 / 1000`.
        
2.  **Generate H.264 Anchor:**
    
    -   Use `ffmpeg` to compress the test video to **match your model's bitrate**.
        
    -   Command: `ffmpeg -i input.mp4 -b:v <YOUR_BITRATE>k -c:v libx264 h264_output.mp4`
        
3.  **Compare:**
    
    -   Calculate PSNR/SSIM for the `h264_output.mp4`.
        
    -   **Pass Condition:** Your Neural Model should have **higher SSIM** than H.264 at low bitrates (< 50kbps), or at least be comparable.
        

### **C. Hardware Constraints Check**

1.  **Model Size:** Must be < **20 MB** (Float32).
    
2.  **SRAM Usage:** State Buffer must be < **320 KB**.
    

----------

## **6. Submission Checklist**

1.  [ ] **Model Code:** Your class inheriting from `StreamableModel` (TF/Keras).
    
2.  [ ] **Benchmark Table:**
    
    -   Columns: `Model Type`, `PSNR (dB)`, `SSIM`, `SRAM Usage (KB)`.
        
    -   Row 1: Your Neural Model.
        
    -   Row 2: H.264 (FFmpeg baseline).
        
3.  [ ] **Visual Proof:** A "Film Strip" comparing Original vs Reconstructed frames.
    
4.  [ ] **Architecture Diagram:** Drawing showing where the "State" connects in your model.
