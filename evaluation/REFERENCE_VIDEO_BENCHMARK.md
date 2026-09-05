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
| Grading | VGG low-level style similarity | higher | Reference-conditioned color/texture feature statistics are closer |
| Grading | LLM reference-style similarity | higher | An independent blinded vision model rates abstract grading similarity on a normalized 0--1 scale |
| Content | Local-normalised structure correlation | higher | Geometry and visible texture survive the grade |
| Content | Edge-SSIM | higher | Edge layout remains intact after the grade |
| Content | DINOv2 cosine similarity | higher | Learned semantic/structural features remain close to the input |
| Temporal | Cut-masked flow warping error | lower | Output frames remain coherent under source-video optical flow |
| Temporal | Cut-masked edit warping error | lower | The edit does not add motion-compensated flicker |
| Temporal | Cut-masked temporal style drift | lower | The applied tonal/color signature does not jump between adjacent frames |
| Quality | MUSIQ | higher | Generic no-reference image quality on sampled frames |
| Artifacts | New shadow clipping | lower | The method does not introduce crushed black pixels |
| Artifacts | New highlight clipping | lower | The method does not introduce clipped white pixels |

Structure correlation removes local luminance and contrast before comparison.
This avoids the earlier gradient-magnitude SSIM failure, which penalised
legitimate exposure and tone-curve changes even when geometry was unchanged.

All temporal metrics exclude detected shot boundaries. Warping error is still
limited by optical-flow accuracy, so temporal transform drift is reported
beside it rather than hidden inside a composite score. Transform drift fits a
small global color transform to each input/output frame pair and measures how
much those fitted coefficients jump between adjacent frames.

The learned metrics are enabled with `--learned-metrics` and use eight sampled
frames by default. Install `requirements-evaluation.txt` before the first run;
the DINO and MUSIQ weights are downloaded once and cached. Pass the
`vgg_normalised.pth` distributed with SA-LUT using `--style-vgg-weights`.

The VGG grading metric compares first- and second-order statistics from
low-level feature maps, following the feature-statistics/style-loss family used
by photorealistic style-transfer work. It remains a cross-content proxy, so the
blinded reference-style win rate is authoritative.

LLM reference-style similarity is the anonymized judge's 1--5
`reference_style_match` rating normalized to 0--1. The judge sees the target,
reference, and eight ordered output frames, and is explicitly instructed not to
reward edit magnitude or raw object-color coincidence. It derives the score as
the unweighted mean of nine concrete style fields: deep-shadow black level,
shadow chroma, midtone luminance, midtone palette, highlight roll-off, neutral
axis/temperature, palette hierarchy, saturation hierarchy, and local
contrast/depth. Content preservation, temporal consistency, artifacts, and
overall preference remain separate diagnostics. Record the judge model and
prompt protocol, and report this score separately from VGG similarity.

## Reference-affinity editor in the API pool

CAP-VSTNet combines a reversible residual representation, whitening/coloring
of feature statistics, and a Matting-Laplacian training constraint to preserve
feature and pixel affinity. ShotAgent does not load CAP-VSTNet weights or use
its output as a prior. Its API pool instead contains a reference-affinity
editor that receives a deterministic profile measured directly from the two
input videos:

- seven lightness quantiles and five explicit tone zones;
- shadow/midtone/highlight Lab bias and chroma;
- five dominant palette clusters with area weights;
- chroma quantiles and temporal lightness/chroma dispersion;
- target-minus-reference deltas without cross-scene pixel correspondence.

The editor must return a tone-zone plan, palette plan, semantic
correspondences, and an affinity-preservation plan. Two other API editors
independently propose a stronger style candidate and a detail-preserving
candidate. The existing metric and visual critics select among them. This
adapts the useful affinity/statistics ideas to the pool while retaining the
ShotAgent inference contract: target video plus reference video only.

MUSIQ was trained as a generic image-quality predictor. A cinematic grade may
intentionally use dark exposure, low contrast, grain or restrained saturation,
so this score is reported separately and must not override the blinded
style/overall-preference result.

For ShotAgent candidate selection, non-grading metrics use an approximately
1% relative tolerance to absorb codec and sampled-model noise. Among candidates
inside that preservation envelope, select the candidate using VGG and LLM
reference-style similarity. On official demo 1, the selected API-Pool result
keeps every non-grading metric within that envelope; structure, flow error,
edit-warp error, transform drift, and clipping are unchanged or improved.

## Official demo 1: current objective table

| Paper method | VGG sim. ↑ | LLM sim. ↑ | Structure ↑ | DINO ↑ | Edge-SSIM ↑ | Flow warp ↓ | Edit warp ↓ | Drift ↓ | New clip ↓ | MUSIQ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SA-LUT: Spatial Adaptive 4D LUT (ICCV 2025) | 0.9032 | 0.2417 | 0.8575 | 0.7526 | 0.7590 | 0.00187 | 0.00193 | 0.01601 | 53.53% | 35.49 |
| NLUT: Neural 3D LUT for Video PST (2023) | 0.9544 | 0.8056 | 0.9558 | 0.9442 | 0.7537 | 0.00223 | 0.00182 | 0.01348 | 0.00% | 34.57 |
| CAP-VSTNet (CVPR 2023) | **0.9701** | **0.8694** | 0.8805 | 0.6806 | 0.7645 | 0.00189 | 0.00218 | 0.01743 | 0.00% | **39.59** |
| CanonCGT (CVPR 2026) | 0.9164 | 0.3611 | 0.9819 | 0.9774 | 0.8320 | 0.00201 | 0.00184 | 0.01546 | 1.83% | 30.37 |
| **ShotAgent API Editor Pool** | 0.9254 | 0.5861 | **0.9878** | **0.9798** | **0.9183** | 0.00217 | **0.00157** | **0.01145** | **0.00%** | 37.89 |

This is a diagnostic table rather than a single-score ranking. CAP-VSTNet and
NLUT move farther toward the reference proxy, while ShotAgent leads the four
graded methods on structure, DINO, Edge-SSIM, edit stability, transform drift,
and clipping. Formal style ranking still comes from the blinded video review.

## Metrics excluded from the main table

Do not use raw RGB/Lab histogram distance, histogram correlation, Lab EMD,
CLIP-T, CLIP directional similarity, CLIP-IQA, input-to-output Delta-E,
output-to-reference SSIM/LPIPS/Delta-E, or changed-pixel fraction in this
benchmark. With unrelated target/reference
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
  --learned-metrics --learned-frame-count 8 \
  --style-vgg-weights /path/to/vgg_normalised.pth
```

Run the anonymized LLM style review and attach its normalized similarity to the
same report with:

```bash
python -m evaluation.blind_video_judge \
  --review-dir outputs/reference_video_eval/demo1_results_modern \
  --sample official-v1-to-v2 \
  --output outputs/reference_video_eval/demo1_results_modern/mllm_reference_style_review.json \
  --benchmark-report outputs/reference_video_eval/demo1_results_modern/report.json
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

The report schema is `reference-video-grade-benchmark/v5-no-gt`. It explicitly
records that no composite score exists.
