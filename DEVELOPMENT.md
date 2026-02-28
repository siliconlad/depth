# Development Guide: Self-Supervised Monocular Depth Estimation

This project implements self-supervised monocular depth estimation in tinygrad,
trained using stereo image pairs. The model learns to predict depth from a single
image by learning to reconstruct one stereo view from the other.

## Your Setup

- **CPU:** AMD Threadripper
- **GPU:** NVIDIA RTX 4090 (24 GB VRAM)
- **Camera:** StereoLabs ZED stereo camera
- **Framework:** tinygrad
- **Experiment tracking:** track-sdk (logs to MCAP files)

## Project Structure

```
src/depth/
    models/
        encoder.py        # Feature extraction backbone (e.g., ResNet18)
        decoder.py        # Disparity prediction head
    losses/
        photometric.py    # Image reconstruction loss (SSIM + L1)
        smoothness.py     # Edge-aware disparity smoothness regularization
    data/
        stereo.py         # Stereo pair data loading
        transforms.py     # Image augmentation
    training/
        trainer.py        # Training loop, optimizer setup, logging
    utils/
        visualization.py  # Depth colorization, disparity-to-depth conversion
```

---

## Reading List

Work through these roughly in order. You don't need to finish everything before
starting to code — the phases below indicate when each reading becomes relevant.

### Foundational ML (before Phase 1)

1. **Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation: building micrograd"**
   https://www.youtube.com/watch?v=VMj-3S1tku0
   (~2.5 hours) You build a tiny autograd engine from scratch. This is the single
   most important prerequisite. After this you'll understand what tinygrad is doing
   under the hood.

2. **3Blue1Brown — "Neural Networks" playlist**
   https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
   (4 videos, ~1 hour) Visual intuition for what neural networks compute.

3. **Andrej Karpathy — "Building makemore" (parts 1-3)**
   https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
   Covers loss functions, optimization, batch normalization.

### Convolutional Neural Networks (before Phase 1)

4. **Stanford CS231n — Lectures 5-9**
   Available on YouTube. Focus on: convolution operation, pooling, ResNet
   architecture, training techniques (learning rate schedules, data augmentation).

5. **"Deep Residual Learning for Image Recognition" (He et al., 2015)**
   https://arxiv.org/abs/1512.03385
   Your encoder is a ResNet. Understand what skip connections do and why they help.

### Self-Supervised Depth (before Phase 2)

6. **"Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue" (Garg et al., 2016)**
   https://arxiv.org/abs/1603.04992
   The foundational idea. Explains why you can learn depth from stereo pairs
   without ground truth labels. Read this to understand the *geometry* of the
   approach.

7. **"Unsupervised Monocular Depth Estimation with Left-Right Consistency" (Godard et al., 2017) — Monodepth**
   https://arxiv.org/abs/1609.03677
   Adds left-right disparity consistency. Clear writing. This is the architecture
   you'll implement first.

8. **"Digging Into Self-Supervised Monocular Depth Estimation" (Godard et al., 2019) — Monodepth2**
   https://arxiv.org/abs/1806.01260
   Key improvements: minimum reprojection loss, auto-masking stationary pixels,
   full-resolution multi-scale. This is what Phase 3 targets.

### Advanced / SOTA (Phase 5)

9. **"3D Packing for Self-Supervised Monocular Depth Estimation" (Guizilini et al., 2020) — PackNet-SfM**
   https://arxiv.org/abs/1905.02693

10. **"HR-Depth: High Resolution Self-Supervised Monocular Depth Estimation" (Lyu et al., 2021)**
    https://arxiv.org/abs/2012.07356

11. **"Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation" (Zhang et al., 2023)**
    https://arxiv.org/abs/2211.13202

12. **"Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data" (Yang et al., 2024)**
    https://arxiv.org/abs/2401.10891
    Current SOTA direction. Uses a DINOv2 ViT encoder with massive pre-training.

### tinygrad

13. **tinygrad documentation:** https://docs.tinygrad.org/
14. **tinygrad examples on GitHub:** https://github.com/tinygrad/tinygrad/tree/master/examples
    Study any CNN examples. The library is small enough to read the source when stuck.

---

## The Core Idea

A stereo camera gives you two images of the same scene from slightly different
viewpoints (left and right, separated by a known baseline distance). For any
pixel in the left image, the corresponding pixel in the right image is shifted
horizontally by an amount called **disparity**. Disparity is inversely
proportional to depth:

```
depth = (baseline * focal_length) / disparity
```

The self-supervised trick: train a neural network to predict disparity from the
**left image alone**, then use that disparity to warp the **right image** into
the left viewpoint. If the predicted disparity is correct, the warped right image
will match the actual left image. The photometric difference between them is
your loss — no labels needed.

At inference time, you throw away the right image entirely. The network has
learned to predict depth from a single image.

---

## Phase 1: Get a Single Pair Working

**Goal:** Overfit on one stereo pair to prove the pipeline works end-to-end.

### 1.1 Record a stereo pair from your ZED camera

- Use the ZED SDK to capture **rectified** left/right image pairs
- Rectification is critical — it guarantees that corresponding points lie on the
  same horizontal scanline, so disparity is purely horizontal
- Even a single pair is enough to test with
- Save the camera calibration: you need the **baseline** (distance between
  cameras, in meters) and **focal length** (in pixels)
- Store images as:
  ```
  data/
      left/
          000000.png
      right/
          000000.png
  ```

### 1.2 Implement image loading (`data/stereo.py`)

- Load a PNG from disk into a tinygrad Tensor
- Pillow is already installed (it's a dependency of both tinygrad and track-sdk)
- Normalize pixel values to [0, 1] float
- Output shape should be (C, H, W) — channels first, as tinygrad expects for Conv2d
- Implement a function that lists and loads stereo pairs from the directory
  structure above

### 1.3 Implement the encoder (`models/encoder.py`)

- Build a ResNet18 encoder
- The encoder takes an image (B, 3, H, W) and returns feature maps at multiple
  scales. You need features at 5 scales: after the stem (1/2 resolution), and
  after each of the 4 residual layer groups (1/4, 1/8, 1/16, 1/32)
- A ResNet18 has the following structure:
  - Stem: 7x7 conv (stride 2) -> BatchNorm -> ReLU -> 3x3 max pool (stride 2)
  - Layer 1: 2 residual blocks, 64 channels
  - Layer 2: 2 residual blocks, 128 channels, first block stride 2
  - Layer 3: 2 residual blocks, 256 channels, first block stride 2
  - Layer 4: 2 residual blocks, 512 channels, first block stride 2
- Each residual block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> add skip -> ReLU
- When the channel count or spatial size changes, the skip connection needs a
  1x1 conv + BN to match dimensions
- tinygrad has `nn.Conv2d` and `nn.BatchNorm` — check the tinygrad docs for
  their exact API

### 1.4 Implement the decoder (`models/decoder.py`)

- Takes the list of encoder feature maps and produces disparity maps
- Architecture: at each level, upsample by 2x (nearest neighbor), concatenate
  with the corresponding encoder skip connection, then apply conv blocks
- Produce disparity output heads (1-channel conv -> sigmoid) at 4 scales
- Sigmoid output means disparity is in [0, 1] — you'll scale it to a useful
  range later (see `utils/visualization.py` for the disp-to-depth conversion)

### 1.5 Implement the photometric loss (`losses/photometric.py`)

- This is a weighted combination of L1 loss and SSIM (Structural Similarity):
  `loss = alpha * SSIM_loss + (1 - alpha) * L1_loss` where alpha is typically 0.85
- SSIM compares local patches (3x3 windows) between two images using their
  means, variances, and covariance. You can compute the local statistics using
  average pooling with kernel_size=3, stride=1, padding=1
- SSIM formula: `((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))`
  where C1 = 0.01^2, C2 = 0.03^2
- The loss is `(1 - SSIM) / 2`, clipped to [0, 1]

### 1.6 Implement the smoothness loss (`losses/smoothness.py`)

- Encourages the disparity map to be locally smooth, but allows sharp edges
  where the image has edges
- Compute disparity gradients in x and y directions (just pixel differences)
- Compute image gradients in x and y directions
- Weight the disparity gradients by `exp(-|image_gradient|)` — this suppresses
  the smoothness penalty at image edges
- Normalize disparity by its mean before computing gradients (makes the loss
  scale-invariant)

### 1.7 Implement differentiable bilinear sampling (`training/trainer.py`)

This is the hardest piece of plumbing and the heart of the self-supervised
training signal. You need to implement a `grid_sample` equivalent.

Given the right image and a predicted disparity map, you need to:

1. Create a meshgrid of pixel coordinates for the output (left) image:
   - `x_coords`: 0, 1, 2, ..., W-1 (repeated for each row)
   - `y_coords`: 0, 1, 2, ..., H-1 (repeated for each column)
2. Shift x-coordinates by the predicted disparity:
   `x_source = x_coords - disparity` (pixels in the left image come from
   further right in the right image, so we subtract)
3. Normalize coordinates to [-1, 1] range (required for bilinear interpolation)
4. For each output pixel, bilinearly sample from the right image at the
   (x_source, y_source) location
5. Crucially: this entire operation must be differentiable so gradients flow
   back through the disparity prediction

Bilinear interpolation for a point (x, y) between four pixel neighbors:
```
top_left     = image[floor(y), floor(x)]
top_right    = image[floor(y), ceil(x)]
bottom_left  = image[ceil(y), floor(x)]
bottom_right = image[ceil(y), ceil(x)]

wx = x - floor(x)    # horizontal weight
wy = y - floor(y)    # vertical weight

result = top_left * (1-wx)*(1-wy) + top_right * wx*(1-wy)
       + bottom_left * (1-wx)*wy  + bottom_right * wx*wy
```

This is entirely composed of multiplications and additions on tensors, so
tinygrad can differentiate through it automatically.

### 1.8 Wire up the training loop

- Forward pass: left image -> encoder -> decoder -> disparity
- Warp right image to left viewpoint using predicted disparity
- Compute photometric loss between warped image and actual left image
- Compute smoothness loss on the disparity map
- Total loss = photometric_loss + 0.001 * smoothness_loss
- Use Adam optimizer, learning rate 1e-4
- Log the loss with track-sdk after each step

### 1.9 Validation

- Feed the same stereo pair repeatedly for ~1000 iterations
- The loss should decrease steadily
- The predicted disparity should start to look like a rough depth map
  (close objects = high disparity = bright, far objects = dark)
- The warped right image should closely match the actual left image

---

## Phase 2: Train on KITTI

**Goal:** Train on a real dataset and evaluate against published benchmarks.

### 2.1 Get the dataset

**KITTI Raw Data:**
- Download from: https://www.cvlibs.net/datasets/kitti/raw_data.php
- You want the **"synced+rectified data"** — these are stereo pairs with
  known camera calibration
- Full dataset is ~175 GB. Start with just the "city" category (~25 GB) to
  iterate faster.
- The standard train/test split is the **Eigen split**. File lists are at:
  https://github.com/mrharicot/monodepth#kitti
- KITTI also provides sparse LiDAR ground truth for evaluation — you do NOT
  use this for training, only for measuring accuracy

**KITTI directory structure:**
```
kitti_raw/
    2011_09_26/
        2011_09_26_drive_0001_sync/
            image_02/data/    # left camera images
            image_03/data/    # right camera images
        calib_cam_to_cam.txt  # camera calibration (baseline, focal length)
    ...
```

### 2.2 Write a KITTI data loader

- Parse the Eigen split file lists to determine train/val/test images
- Read calibration files to extract baseline and focal length for each date
- Load stereo pairs (image_02 = left, image_03 = right)
- Resize to 640x192 (the standard training resolution — fits well on 4090)

### 2.3 Add data augmentation (`data/transforms.py`)

- **Random horizontal flip:** flip both images AND swap left/right. This
  doubles the effective data. Remember to also flip the disparity convention.
- **Color jitter:** random brightness, contrast, saturation, hue perturbations.
  Apply the SAME perturbation to both left and right images.
- **Random crop** (optional at this stage)

### 2.4 Train

- Batch size 12 at 640x192 (should use ~12-16 GB VRAM on your 4090)
- Adam optimizer, lr=1e-4 for 20 epochs, drop to 1e-5 for the last 5 epochs
- Use track-sdk to log:
  - Loss curves (photometric, smoothness, total)
  - Sample disparity predictions every N steps (use `log_image`)
  - Learning rate
  - Epoch/step counters as metadata

### 2.5 Evaluate

Standard evaluation protocol (Eigen et al.):

- Predict depth for each image in the Eigen test split (697 images)
- Cap predicted depth at 80m
- Use Garg's crop (remove top/bottom borders where sky/car hood dominate)
- Compute these metrics against LiDAR ground truth:
  - **AbsRel:** mean(|d_pred - d_gt| / d_gt) — lower is better
  - **SqRel:** mean((d_pred - d_gt)^2 / d_gt) — lower is better
  - **RMSE:** sqrt(mean((d_pred - d_gt)^2)) — lower is better
  - **delta < 1.25:** % of pixels where max(d_pred/d_gt, d_gt/d_pred) < 1.25 — higher is better

**Target:** AbsRel < 0.130 (roughly Monodepth1 level)

---

## Phase 3: Monodepth2 Improvements

**Goal:** Implement the three key innovations from Monodepth2 (Godard et al., 2019).
Read the paper carefully before starting this phase.

### 3.1 Minimum reprojection loss

Instead of averaging the photometric loss from multiple source images, take the
**per-pixel minimum**. This handles occlusion — pixels visible in the left image
but not the right will have high loss; taking the minimum lets other sources
(or scales) provide the training signal instead.

For stereo-only training, this means: compute the photometric loss for the
warped right-to-left AND the warped left-to-right, then take the per-pixel
minimum.

### 3.2 Auto-masking of stationary pixels

Some pixels have the same appearance in both views regardless of depth (e.g.,
sky, very distant objects, textureless regions). These provide no useful gradient.

The fix: also compute the photometric loss of the **unwarped** source image
(the "identity" warp). If the identity loss is lower than the predicted warp
loss for a pixel, mask that pixel out — the network can't do better than
not warping at all, so there's nothing to learn there.

Formally: `mask = (reprojection_loss < identity_loss).float()`

### 3.3 Full-resolution multi-scale

In Phase 2, you might compute the photometric loss at each decoder scale
separately (downscaling the images to match). This causes texture-copy artifacts.

Instead: upsample ALL disparity predictions to the full input resolution before
computing the photometric loss. The loss is always computed at full resolution.
The multi-scale disparity outputs are only used for this upsampling, not for
separate loss computation.

### 3.4 Target

AbsRel < 0.110 on KITTI Eigen.

---

## Phase 4: Your Own Stereo Data

**Goal:** Make the model work well in your room using your ZED camera.

### 4.1 Recording data

- Use the ZED SDK to record walking sequences through your room
- Capture **rectified** stereo pairs (the SDK does this for you)
- Record at the camera's native resolution
- Save camera calibration (baseline + focal length)
- Record 5-10 minute sequences at different times of day for lighting variation
- Move slowly and smoothly — fast motion causes blur
- Cover the full environment: multiple rooms, angles, object arrangements
- Aim for 10,000-50,000 frames total

### 4.2 Data organization

```
zed_data/
    sequence_001/
        left/
            000000.png
            000001.png
            ...
        right/
            000000.png
            000001.png
            ...
        calibration.json    # {"baseline": 0.12, "focal_length": 700.0}
    sequence_002/
        ...
```

### 4.3 Training strategy

Two options:

**Option A — Fine-tune (recommended to start):**
Take your best KITTI-trained model from Phase 3. Fine-tune on your ZED data
for 5-10 epochs at a low learning rate (1e-5). This transfers the general
depth understanding from KITTI and adapts it to your camera and environment.

**Option B — Train from scratch:**
Train on your ZED data alone. Needs more data (30k+ frames) but gives a model
perfectly tuned to your specific setup.

### 4.4 Indoor-specific challenges

- **Textureless regions** (white walls, ceilings): photometric loss has no
  signal here. Auto-masking from Phase 3 helps. The smoothness loss fills in
  these areas by propagating from nearby textured regions.
- **Reflective surfaces** (mirrors, screens, glossy furniture): these violate
  the brightness constancy assumption. No clean fix — accept noisy depth on
  these surfaces.
- **Close range**: indoor depth range is ~0.3-10m vs KITTI's ~1-80m. Update
  your min_depth/max_depth parameters for disparity-to-depth conversion.
- **Lighting variation**: indoor lighting changes drastically. Color augmentation
  during training is important.

### 4.5 Qualitative evaluation

Since you won't have LiDAR ground truth for your room:
- Visually inspect depth maps — do edges align with object boundaries?
- Measure known distances (e.g., distance to a wall) with the ZED's depth
  sensor and compare to your model's predictions
- Log predictions with track-sdk for side-by-side comparison

---

## Phase 5: Toward State-of-the-Art

**Goal:** Push accuracy as far as your 4090 allows.

### 5.1 Better encoder

- **ResNet50** instead of ResNet18: more capacity, deeper features. Still
  trains fine on a 4090 at 640x192 with batch size 8-10.
- **Pre-trained weights**: loading ImageNet-pretrained weights into the encoder
  gives a massive head start. You'll need to port weights from another format
  (e.g., PyTorch .pth) into tinygrad tensors. This is fiddly but worth it —
  pre-training is the single biggest accuracy boost.
- **Vision Transformer (ViT) encoder**: the SOTA direction (Depth Anything uses
  DINOv2-ViT). Much more complex to implement in tinygrad, but very powerful.

### 5.2 Better decoder

- Replace nearest-neighbor upsampling with **pixel-shuffle** (sub-pixel
  convolution) or learned upsampling
- Add **attention gates** at skip connections to help the decoder focus on
  relevant encoder features
- Consider the skip connection redesign from HR-Depth for sharper edges

### 5.3 Additional losses

- **Disparity consistency loss:** predict disparity from BOTH left and right
  images, warp one disparity map to the other viewpoint, and penalize
  differences. Enforces geometric consistency.
- **Feature-metric loss** (Shu et al., 2020): compute photometric loss in
  learned feature space instead of pixel space. Helps in textureless regions
  where pixel-level photometric loss gives no gradient.

### 5.4 Data scaling

- Mix KITTI + your ZED data during training (sample from both datasets in
  each batch)
- Train at higher resolution: 1024x320 (still fits on 4090 with batch size 4-6)
- Add more augmentation: random resized crop, elastic deformation

### 5.5 Test-time improvements

- **Horizontal flip averaging:** at inference, predict depth for both the
  original image and its horizontal flip, then average. Free ~2% improvement.
- **Multi-crop averaging:** predict on overlapping crops and stitch together
  for very high resolution output.

### 5.6 Realistic targets

| Encoder | Improvements | KITTI AbsRel |
|---------|-------------|-------------|
| ResNet18 | Monodepth2 | ~0.110 |
| ResNet50 | Monodepth2 + pre-trained | ~0.095 |
| ResNet50 | + consistency + feature-metric | ~0.085 |
| ViT (DINOv2) | Depth Anything style | ~0.065 |

---

## Summary: What You Learn at Each Phase

| Phase | You build | You learn |
|-------|-----------|-----------|
| 1 | End-to-end pipeline on 1 image pair | Autograd, convolutions, how self-supervised depth works geometrically |
| 2 | Full training on KITTI | Data loading, augmentation, evaluation protocols, experiment tracking |
| 3 | Monodepth2 loss improvements | Loss function design, handling edge cases (occlusion, static scenes) |
| 4 | Training on your own stereo data | Domain adaptation, real-world data collection, indoor depth challenges |
| 5 | Architecture improvements | Encoder design, pre-training, attention, advanced loss functions |

---

## Quick Reference: Verifying Your GPU Setup

```bash
# Check that tinygrad sees your GPU
CUDA=1 python -c "from tinygrad import Device; print(Device.DEFAULT)"
# Should print "CUDA"

# Quick smoke test
CUDA=1 python -c "from tinygrad import Tensor; print((Tensor.randn(2,3) @ Tensor.randn(3,2)).numpy())"
```

## Quick Reference: track-sdk Usage

```python
from track_sdk import Logger

logger = Logger("experiment_name")

# During training
logger.log_image("depth_prediction", depth_array)
logger.add_metadata({"epoch": epoch, "loss": loss_value, "lr": lr})
logger.add_attachment("model_weights.safetensors", model_path)
```

Logs are saved as MCAP files with timestamped filenames.
