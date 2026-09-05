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
| Reference color | Normalized Lab histogram EMD | lower | Output and reference marginal Lab distributions become closer |
| Instruction | CLIP-T | higher | Output frames agree with a supplied textual look instruction |
| Style direction | CLIP directional similarity | higher | The target-to-output CLIP change points toward the target-to-reference change |
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
the model weights are downloaded once and cached. CLIP-T is emitted only when
the manifest sample contains an `instruction` string.

Lab EMD is the mean of normalized one-dimensional Wasserstein distances for
the L*, a* and b* marginals. It is useful for detecting whether a method moved
toward the reference palette, but it is affected by different objects and scene
composition. CLIP directional similarity has the same cross-content limitation:
without an ungraded version of the reference video, the target-to-reference
direction also contains semantic change. Both remain auxiliary diagnostics.

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

Do not use raw RGB/Lab histogram distance, histogram correlation,
output-to-reference SSIM/LPIPS/Delta-E, or changed-pixel fraction as the primary
ranking. Those quantities mostly measure which colors and objects happen to
occur in the two scenes. Normalized Lab EMD is retained only as a named
reference-color diagnostic. BRISQUE and NIQE may be included in an appendix but
are not reliable arbiters of intentional cinematic looks.

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

The report schema is `reference-video-grade-benchmark/v3-no-gt`. It explicitly
records that edit magnitude is descriptive and that no composite score exists.
