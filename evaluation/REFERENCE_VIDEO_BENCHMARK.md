# Reference-video grading benchmark (no GT)

This benchmark evaluates a black-box visual contract:

```text
target video + reference video -> graded target video
```

ShotAgent may use an API editor pool and another method may generate a LUT. The
internal representation is outside the quality evaluation. Every method gets
the same target/reference pair and only its rendered RGB video is measured.

## What determines the ranking

There is no aligned target rendition, so PSNR, SSIM, LPIPS and Delta-E against
GT are undefined. The primary ranking is a blinded pairwise review of complete
videos:

1. **Reference-style win rate:** which candidate better transfers the
   reference's contrast, tonal hierarchy, palette and atmosphere?
2. **Overall-preference win rate:** which candidate better balances the
   reference look with content preservation, temporal stability and artifacts?

Report both win rates with bootstrap 95% confidence intervals. Do not combine
them with engineering metrics into one total score. Use at least three human
raters per pair for a formal result. An independently configured MLLM judge can
be reported as a separate development result, not as human preference.

The script writes two ready-to-use forms:

- `blind_pairwise_review.csv`: A/B/Tie decisions for win rates.
- `blind_individual_review.csv`: 1--5 diagnostic ratings for reference match,
  content preservation, temporal consistency and artifact control.

Candidate videos have anonymous codes under `blind_review_media/`. Keep
`blind_review_key.json` away from raters until scoring is complete.

## Objective axes without GT

These metrics diagnose failure modes and never replace the style-match review.

| Axis | Metric | Direction | Interpretation |
|---|---|---:|---|
| Content | Local-normalised structure correlation | higher | Geometry and visible texture survive the grade |
| Content | Edge-SSIM | higher | Edge layout remains intact after the grade |
| Content | DINOv2 cosine similarity | higher | Learned semantic/structural features remain close to the input |
| Temporal | Cut-masked flow warping error | lower | Output frames remain coherent under source-video optical flow |
| Temporal | Cut-masked edit warping error | lower | The edit does not add motion-compensated flicker |
| Temporal | Cut-masked temporal style drift | lower | The applied tonal/color signature does not jump between adjacent frames |
| Quality | MUSIQ | higher | Generic no-reference image quality on sampled frames |
| Quality | CLIP-IQA | higher | CLIP-based no-reference image quality on sampled frames |
| Artifacts | New shadow clipping | lower | The method does not introduce crushed black pixels |
| Artifacts | New highlight clipping | lower | The method does not introduce clipped white pixels |

Structure correlation removes local luminance and contrast before comparison.
This avoids the earlier gradient-magnitude SSIM failure, which penalised
legitimate exposure and tone-curve changes even when geometry was unchanged.

All temporal metrics exclude detected shot boundaries. Warping error is still
limited by optical-flow accuracy, so temporal transform drift is reported
beside it rather than hidden inside a composite score. Transform drift fits a
small Lab affine transform to each input/output frame pair and measures how
much those fitted coefficients jump between adjacent frames.

The learned metrics are enabled with `--learned-metrics` and use eight sampled
frames by default. Install `requirements-evaluation.txt` before the first run;
the model weights are downloaded once and cached.

MUSIQ and CLIP-IQA were trained as generic image-quality predictors. A cinematic
grade may intentionally use dark exposure, low contrast, grain or restrained
saturation, so these scores are reported separately and must not override the
blinded style/overall-preference result.

## Edit magnitude is a descriptor

`edit_magnitude_delta_e00` and the fraction of pixels with Delta-E00 above 2
describe how far the output moved from its input. They have **no better
direction**, no pass threshold and no weight in any score. A small edit can be
correct for a subtle reference; a large edit can be correct for a stylised
reference or wrong because of an excessive cast.

If analysis by strength is useful, define bins after collecting results and
report each method's win rate inside those bins. Never award points merely for
landing in a stronger bin.

## Metrics excluded from the main table

Do not use raw RGB/Lab histogram distance, histogram correlation, Lab EMD,
CLIP-T, CLIP directional similarity, output-to-reference SSIM/LPIPS/Delta-E, or
changed-pixel fraction in this benchmark. With unrelated target/reference
content, these quantities are dominated by scene semantics or color
composition. BRISQUE and NIQE may be included in an appendix but are not
reliable arbiters of intentional cinematic looks.

## Running the benchmark

```bash
python -m evaluation.reference_video_benchmark \
  --manifest outputs/reference_video_eval/official_demos/manifest.json \
  --output-dir outputs/reference_video_eval/official_results_nogt \
  --methods identity global-reinhard global-mkl framewise-reinhard
```

Add any rendered black-box output directory with:

```bash
python -m evaluation.reference_video_benchmark \
  --manifest outputs/reference_video_eval/official_demos/manifest_demo1.json \
  --output-dir outputs/reference_video_eval/demo1_results_nogt \
  --methods identity global-reinhard global-mkl framewise-reinhard \
  --external shotagent-pool=outputs/reference_video_eval/shotagent_visual_outputs
```

Enable learned metrics with:

```bash
uv pip install --python .venv/bin/python -r requirements-evaluation.txt
python -m evaluation.reference_video_benchmark \
  --manifest outputs/reference_video_eval/official_demos/manifest_demo1.json \
  --output-dir outputs/reference_video_eval/demo1_results_perceptual \
  --methods identity global-reinhard global-mkl framewise-reinhard \
  --external shotagent-pool=outputs/reference_video_eval/shotagent_visual_outputs \
  --learned-metrics --learned-frame-count 8
```

## Modern reference-style baselines

Recent image-reference methods can enter the same video protocol by selecting
the temporal middle frame of the reference video as the style image. Methods
that estimate one global 3D LUT also use the middle frame of the target video
during estimation and apply the resulting LUT to every target frame. This
keeps the reference selection fixed and avoids frame-by-frame re-estimation.

Run the official SA-LUT checkpoint with:

```bash
python -m evaluation.run_salut_baseline \
  --repo-dir /path/to/SA-LUT \
  --checkpoint /path/to/SA-LUT.ckpt \
  --target target.mp4 --reference reference.mp4 \
  --output outputs/sa-lut/sample.mp4
```

Run the official NLUT checkpoint and its 40-step test-time fine-tuning with:

```bash
python -m evaluation.run_nlut_baseline \
  --repo-dir /path/to/NLUT \
  --checkpoint /path/to/336999_style_lut.pth \
  --target target.mp4 --reference reference.mp4 \
  --output outputs/nlut/sample.mp4
```

Run the official CAP-VSTNet photorealistic video checkpoint without semantic
masks with:

```bash
python -m evaluation.run_cap_vstnet_baseline \
  --repo-dir /path/to/cap-vstnet \
  --checkpoint /path/to/photo_video.pt \
  --target target.mp4 --reference reference.mp4 \
  --output outputs/cap-vstnet/sample.mp4
```

Run the official CanonCGT self-supervised checkpoint with:

```bash
python -m evaluation.run_canoncgt_baseline \
  --repo-dir /path/to/CanonCGT \
  --checkpoint /path/to/SSL_updated_251111.pth \
  --config /path/to/Stage3_SSL_training_Flickr2K_PPR10K_LSDIR.yaml \
  --target target.mp4 --reference reference.mp4 \
  --output outputs/canoncgt/sample.mp4
```

The SA-LUT and NLUT adapters replace the repositories' old custom interpolation
extensions with equivalent PyTorch implementations. All four adapters load the
published weights and preserve the released inference logic. Record any
resolution or memory adaptation beside the result.

The report schema is `reference-video-grade-benchmark/v4-no-gt`. It explicitly
records that edit magnitude is descriptive and that no composite score exists.
