# Loss Functions

## Photometric Loss

**Purpose:** Measure how well a warped image matches the target image. This is the
primary training signal for self-supervised depth estimation — if the predicted
disparity is correct, warping the right image using that disparity should
reconstruct the left image.

**Formula:**

```
L_photometric = (1 - alpha) * L_L1 + alpha * L_SSIM
```

where `alpha = 0.85` by default, weighting SSIM more heavily than L1.

**Intuition:** L1 alone captures per-pixel differences but is sensitive to small
misalignments and lighting changes. SSIM compares local structure (patches), making
it more robust to brightness shifts and better at preserving edges. Combining them
gives the best of both — SSIM handles structure while L1 provides a clean gradient
signal in smooth regions.

---

### L1 Loss

**Purpose:** Measure the mean absolute pixel-wise difference between two images.

**Formula:**

```
L_L1 = mean(|I_predicted - I_target|)
```

**Intuition:** The simplest possible image comparison. Every pixel contributes
equally. It's less sensitive to outliers than L2 (squared error), which is important
because occluded regions will always have high error and shouldn't dominate the loss.

---

### SSIM Loss

**Purpose:** Measure structural similarity between two images using local statistics,
capturing perceptual differences that per-pixel metrics miss.

**Formula:**

For each pixel, compute statistics over a 3x3 local window using average pooling:

```
mu_x    = avg_pool(x)           -- local mean of x
mu_y    = avg_pool(y)           -- local mean of y
var_x   = avg_pool(x^2) - mu_x^2   -- local variance of x
var_y   = avg_pool(y^2) - mu_y^2   -- local variance of y
cov_xy  = avg_pool(x*y) - mu_x*mu_y  -- local covariance
```

Then the SSIM index at each pixel is:

```
SSIM = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2))
```

where `C1 = 0.01^2` and `C2 = 0.03^2` are small constants for numerical stability.

SSIM = 1 means identical patches. The loss converts this to a minimizable quantity:

```
L_SSIM = mean((1 - SSIM) / 2)
```

**Intuition:** SSIM decomposes the comparison into three factors:
1. **Luminance** (`mu_x` vs `mu_y`): are the patches equally bright?
2. **Contrast** (`var_x` vs `var_y`): do they have similar dynamic range?
3. **Structure** (`cov_xy`): are the intensity patterns correlated?

This mirrors how human vision works — we notice structural changes (edges shifting,
textures warping) more than uniform brightness changes. By comparing patches rather
than individual pixels, SSIM is robust to small intensity shifts that would fool L1.

---

## Smoothness Loss

**Purpose:** Regularize the predicted disparity map to be locally smooth, preventing
noisy or discontinuous depth predictions, while still allowing sharp depth edges at
object boundaries.

**Formula:**

First, normalize the disparity map by its mean to make the loss scale-invariant:

```
d_norm = d / mean(d)
```

Then compute disparity and image gradients via finite differences:

```
grad_d_x = |d_norm[:, :, :, 1:] - d_norm[:, :, :, :-1]|
grad_d_y = |d_norm[:, :, 1:, :] - d_norm[:, :, :-1, :]|
grad_I_x = |I[:, :, :, 1:] - I[:, :, :, :-1]|
grad_I_y = |I[:, :, 1:, :] - I[:, :, :-1, :]|
```

Weight the disparity gradients by the negative exponential of the image gradients:

```
L_smoothness = mean(grad_d_x * exp(-grad_I_x) + grad_d_y * exp(-grad_I_y)) / 2
```

**Intuition:** Without this loss, the network could predict noisy, jagged depth maps
that still produce low photometric error (especially in textureless regions where
the photometric loss has no signal). The smoothness loss penalizes large disparity
gradients, encouraging flat surfaces to have uniform depth.

The key insight is the **edge-aware weighting**: `exp(-|grad_I|)`. Where the image
has a strong edge (large gradient), the weight drops to ~0, so the disparity is
free to change sharply — this is exactly where depth discontinuities occur (object
boundaries). Where the image is smooth (small gradient), the weight stays ~1, and
the disparity is strongly encouraged to be smooth too.

The mean-normalization ensures the loss doesn't just push all disparity values toward
zero. By dividing by the mean, only the relative spatial variation matters, not the
absolute scale.
