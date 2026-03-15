# ResNet18 Encoder

## References

- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385) —
  the original ResNet paper. Introduces skip connections and the residual block.
- [Digging Into Self-Supervised Monocular Depth Estimation (Godard et al., 2019)](https://arxiv.org/abs/1806.01260) —
  Monodepth2, which uses a ResNet18 encoder for depth estimation.

## Purpose

The encoder extracts features from an input image at multiple spatial scales. These
multi-scale features capture both fine details (edges, textures) at high resolution
and semantic understanding (objects, scene layout) at low resolution. The decoder
uses all of these scales to predict depth.

## Architecture Outline

For an input image of shape `(B, 3, H, W)`:

```
Input (B, 3, H, W)
  |
  v
Stem: Conv2d(3, 64, 7x7, stride=2, pad=3) -> BN -> ReLU
  |
  +--> layer_0 output (B, 64, H/2, W/2)    <-- returned to decoder
  |
MaxPool(3x3, stride=2, pad=1)
  |
  v (B, 64, H/4, W/4)
  |
Layer 1: ResidualBlock(64, 64) x2
  |
  +--> layer_1 output (B, 64, H/4, W/4)    <-- returned to decoder
  |
Layer 2: ResidualBlock(64, 128, stride=2) + ResidualBlock(128, 128)
  |
  +--> layer_2 output (B, 128, H/8, W/8)   <-- returned to decoder
  |
Layer 3: ResidualBlock(128, 256, stride=2) + ResidualBlock(256, 256)
  |
  +--> layer_3 output (B, 256, H/16, W/16) <-- returned to decoder
  |
Layer 4: ResidualBlock(256, 512, stride=2) + ResidualBlock(512, 512)
  |
  +--> layer_4 output (B, 512, H/32, W/32) <-- returned to decoder
```

The encoder returns 5 feature maps: `(layer_0, layer_1, layer_2, layer_3, layer_4)`.

Note that `layer_0` is captured **before** the max pool. This gives the decoder a
skip connection at 1/2 resolution, which is important for reconstructing fine detail
in the final disparity map.

## Components

### Stem

The stem rapidly reduces spatial resolution while expanding from 3 channels (RGB) to
64 feature channels. The large 7x7 kernel gives each neuron a wide receptive field
right from the start, capturing low-frequency structure (broad gradients, large
edges) that smaller kernels would miss.

- `Conv2d(3, 64, 7x7, stride=2, padding=3)`: halves spatial dims, expands channels
- `BatchNorm(64)`: normalizes activations for stable training
- `ReLU`: non-linearity
- `MaxPool(3x3, stride=2, padding=1)`: halves spatial dims again, provides
  translational invariance

After the stem, the feature map is at 1/4 of the input resolution.

### Residual Block

The core building block. Each block applies two 3x3 convolutions with batch
normalization, then adds the input back to the output (the "skip connection"):

```
x ----+
|     |
v     |
Conv2d(3x3) -> BN -> ReLU    |
|     |
v     |
Conv2d(3x3) -> BN            |
|     |
v     |
(+) <-+
|
v
ReLU
```

When the block needs to change the number of channels or the spatial resolution
(stride > 1), the skip connection goes through a 1x1 convolution + BN to match
dimensions:

```
x ----------+
|           |
v           v
Conv2d(3x3, stride) -> BN -> ReLU    Conv2d(1x1, stride) -> BN
|           |
v           |
Conv2d(3x3) -> BN                    |
|           |
v           |
(+) <-------+
|
v
ReLU
```

**Why skip connections work:** In a plain deep network, gradients must flow through
every layer during backpropagation, and they tend to vanish or explode. The skip
connection provides a "gradient highway" — gradients can flow directly from the loss
back to earlier layers through the addition. This makes it possible to train much
deeper networks. It also means each block only needs to learn the *residual* (the
difference from the identity), which is typically a small correction and easier to
learn than the full mapping.

### Residual Layer

A layer groups two residual blocks. The first block may downsample (stride=2) and
change channels. The second block always preserves dimensions.

## Parameter Count

| Component | Parameters |
|---|---|
| **Stem** | |
| Conv2d(3, 64, 7x7) | 3 * 64 * 7 * 7 + 64 = **9,472** |
| BatchNorm(64) | 64 * 2 = **128** |
| **Layer 1** (64 -> 64, no downsample) | |
| Block 1: 2x Conv2d(64, 64, 3x3) + 2x BN | 2 * (64 * 64 * 9 + 64) + 2 * 128 = **74,112** |
| Block 2: same | **74,112** |
| **Layer 2** (64 -> 128, stride=2) | |
| Block 1: Conv2d(64, 128, 3x3) + Conv2d(128, 128, 3x3) + 2x BN + projection | **230,144** |
| Block 2: 2x Conv2d(128, 128, 3x3) + 2x BN | **295,424** |
| **Layer 3** (128 -> 256, stride=2) | |
| Block 1: Conv2d(128, 256, 3x3) + Conv2d(256, 256, 3x3) + 2x BN + projection | **920,064** |
| Block 2: 2x Conv2d(256, 256, 3x3) + 2x BN | **1,180,672** |
| **Layer 4** (256 -> 512, stride=2) | |
| Block 1: Conv2d(256, 512, 3x3) + Conv2d(512, 512, 3x3) + 2x BN + projection | **3,676,160** |
| Block 2: 2x Conv2d(512, 512, 3x3) + 2x BN | **4,720,640** |
| | |
| **Total** | **~11.2M parameters** |

## Intuition

Think of the encoder as a funnel that progressively compresses spatial information
into semantic information:

- **layer_0 (1/2 res, 64ch):** Low-level features — edges, corners, color gradients.
  The large 7x7 stem kernel acts like a collection of edge detectors at various
  orientations.

- **layer_1 (1/4 res, 64ch):** Combines edges into slightly more complex patterns —
  textures, simple shapes, repeated structures.

- **layer_2 (1/8 res, 128ch):** Parts of objects — wheels, window frames, road
  markings. The increased channel count allows more diverse feature types.

- **layer_3 (1/16 res, 256ch):** Whole objects or large object parts. At this scale,
  the receptive field is large enough to "see" significant portions of the scene.

- **layer_4 (1/32 res, 512ch):** Scene-level understanding — spatial layout, relative
  object positions, scene category. Highest abstraction, lowest spatial detail.

For depth estimation, all scales matter. Fine scales tell you exactly where object
edges are (important for sharp depth boundaries). Coarse scales tell you the overall
scene structure (important for getting the right depth ordering — which objects are
in front of which).

# Disparity Decoder

## References

- [Unsupervised Monocular Depth Estimation with Left-Right Consistency (Godard et al., 2017)](https://arxiv.org/abs/1609.03677) —
  Monodepth. Introduces the U-Net style encoder-decoder for self-supervised depth
  with multi-scale disparity outputs.
- [Digging Into Self-Supervised Monocular Depth Estimation (Godard et al., 2019)](https://arxiv.org/abs/1806.01260) —
  Monodepth2. Refines the decoder with nearest-neighbor upsampling (instead of
  transposed convolutions) and reflect padding to reduce artifacts.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597) —
  the original U-Net architecture that established the encoder-decoder with skip
  connections pattern.

## Purpose

The decoder takes the multi-scale encoder features and progressively upsamples them
back to the input resolution, producing a disparity map at each scale. The disparity
map is what gets converted to depth via `depth = baseline * focal_length / disparity`.

## Architecture Outline

The decoder mirrors the encoder: it goes from the deepest (most compressed) features
back up to full resolution. At each stage, it combines upsampled features with the
corresponding encoder skip connection to recover spatial detail.

```
layer_4 (B, 512, H/32, W/32)
  |
  v
Stage 1: Conv(512->256) -> ELU -> Upsample 2x -> Cat(layer_3) -> Conv(512->256) -> ELU
  |
  v (B, 256, H/16, W/16)
  |
Stage 2: Conv(256->128) -> ELU -> Upsample 2x -> Cat(layer_2) -> Conv(256->128) -> ELU
  |                                                                   |
  v (B, 128, H/8, W/8)                                               +--> head: Conv(128->1) -> Sigmoid
  |
Stage 3: Conv(128->64) -> ELU -> Upsample 2x -> Cat(layer_1) -> Conv(128->64) -> ELU
  |                                                                  |
  v (B, 64, H/4, W/4)                                               +--> head: Conv(64->1) -> Sigmoid
  |
Stage 4: Conv(64->32) -> ELU -> Upsample 2x -> Cat(layer_0) -> Conv(96->32) -> ELU
  |                                                                  |
  v (B, 32, H/2, W/2)                                               +--> head: Conv(32->1) -> Sigmoid
  |
Stage 5: Conv(32->16) -> ELU -> Upsample 2x -> Conv(16->16) -> ELU
  |
  +--> head: Conv(16->1) -> Sigmoid
  |
  v (B, 1, H, W) -- full resolution disparity
```

The decoder returns 4 disparity maps at different scales:
`(full_res, half_res, quarter_res, eighth_res)`.

## Components

### Decoder Stage

Each stage follows the same pattern:

1. **Channel reduction conv (conv_a):** 3x3 convolution that halves the channel
   count, followed by ELU activation. This compresses the features before upsampling.

2. **Nearest-neighbor upsample:** Doubles the spatial dimensions by repeating each
   pixel in a 2x2 block. This is preferred over transposed convolutions because
   transposed convolutions produce
   [checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/).

3. **Concatenation with skip connection:** The upsampled features are concatenated
   (channel-wise) with the encoder features at the same resolution. This is the key
   idea from U-Net — the encoder features preserve the fine spatial detail that was
   lost during downsampling.

4. **Fusion conv (conv_b):** 3x3 convolution that combines the concatenated features,
   followed by ELU activation. This learns to merge the two information sources:
   high-level context from the decoder path and fine spatial detail from the encoder.

Note on stage 4: `layer_0` has 64 channels while the upsampled features have 32,
giving 96 channels after concatenation (unlike the symmetric splits in earlier
stages).

Stage 5 has no skip connection — there are no encoder features at full resolution.
It simply upsamples and refines.

### Reflect Padding

All convolutions use manual reflect padding (`pad((1,1,1,1), mode="reflect")`)
instead of zero padding. Zero padding creates a visible border effect — the edge
pixels always see zeros on one side, biasing their activations. Reflect padding
mirrors the image at the boundary, producing more natural results at image edges.
This matters for depth estimation because depth at image borders should continue
smoothly, not snap to an artificial value.

### Disparity Heads

At scales 2 through 5, a separate 3x3 convolution maps features to a single-channel
disparity prediction, followed by sigmoid to constrain the output to [0, 1]. These
multi-scale outputs serve two purposes:

1. **Multi-scale supervision:** during training, the loss can be computed at each
   scale, providing gradient signal at multiple resolutions.
2. **Coarse-to-fine refinement:** early scales capture the rough scene layout while
   later scales refine edges and details.

### Activation: ELU vs ReLU

The decoder uses ELU (Exponential Linear Unit) instead of the encoder's ReLU. ELU
allows small negative outputs (`alpha * (exp(x) - 1)` for `x < 0`), which gives a
non-zero gradient for negative inputs. This helps in the decoder where features need
to be refined rather than just detected — the smooth negative region produces better
gradients for the smooth surfaces common in depth maps.

## Parameter Count

| Component | Parameters |
|---|---|
| **Stage 1** | |
| Conv2d(512, 256, 3x3) | 512 * 256 * 9 + 256 = **1,179,904** |
| Conv2d(512, 256, 3x3) | 512 * 256 * 9 + 256 = **1,179,904** |
| **Stage 2** | |
| Conv2d(256, 128, 3x3) | 256 * 128 * 9 + 128 = **295,040** |
| Conv2d(256, 128, 3x3) | 256 * 128 * 9 + 128 = **295,040** |
| Conv2d(128, 1, 3x3) head | 128 * 1 * 9 + 1 = **1,153** |
| **Stage 3** | |
| Conv2d(128, 64, 3x3) | 128 * 64 * 9 + 64 = **73,792** |
| Conv2d(128, 64, 3x3) | 128 * 64 * 9 + 64 = **73,792** |
| Conv2d(64, 1, 3x3) head | 64 * 1 * 9 + 1 = **577** |
| **Stage 4** | |
| Conv2d(64, 32, 3x3) | 64 * 32 * 9 + 32 = **18,464** |
| Conv2d(96, 32, 3x3) | 96 * 32 * 9 + 32 = **27,680** |
| Conv2d(32, 1, 3x3) head | 32 * 1 * 9 + 1 = **289** |
| **Stage 5** | |
| Conv2d(32, 16, 3x3) | 32 * 16 * 9 + 16 = **4,624** |
| Conv2d(16, 16, 3x3) | 16 * 16 * 9 + 16 = **2,320** |
| Conv2d(16, 1, 3x3) head | 16 * 1 * 9 + 1 = **145** |
| | |
| **Total** | **~3.15M parameters** |

Combined with the encoder (~11.2M), the full model has approximately **14.4M
parameters**.

## Intuition

The decoder answers the question: "given what the encoder understood about the scene,
what is the depth at every pixel?"

The encoder compresses the image into a compact representation that understands
*what* is in the scene but has lost the precise *where*. The decoder's job is to
recover the spatial precision while retaining the semantic understanding.

The skip connections are critical for this. Without them, the decoder would have to
hallucinate all spatial detail from the compressed 1/32 resolution features — like
trying to draw a detailed map from memory. With skip connections, the decoder gets
the original fine-grained spatial information at each scale and only needs to decide
how to *interpret* it as depth. The encoder features tell it "there's a sharp edge
here," and the decoder decides "that edge is a depth boundary between an object at
2m and a wall at 5m."

The multi-scale disparity heads give the training process multiple points where it
can correct the prediction. If the coarsest scale gets the rough layout wrong (e.g.,
predicts a wall where there's a doorway), the gradient signal reaches the deeper
encoder layers more directly than if the loss were only computed at full resolution.
