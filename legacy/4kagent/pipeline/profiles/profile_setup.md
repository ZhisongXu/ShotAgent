We use profile to customize 4KAgent in different restoration tasks. Here we preset the set up of profile in 4KAgent:
```bash
Name: 4KAgent_Profile_Example

PerceptionAgent: llama_vision  # [llama_vision, depictqa]
PerceptionAgent_Seed: 1994     # used when PerceptionAgent is llama_vision
Reflection: hpsv2+metric       # [hpsv2, hpsv2+metric]

Upscale4K: True                # 'Upscale to 4K'
require_sr_size: 300           # if max(H, W) < require_sr_size, append 'super-resolution' for planning, enable when ScaleFactor is null and Upscale4K is false
ScaleFactor: null              # [2, 4, 8, 16], higher priority than Upscale4K
RestoreOption: null            # Explicity sets the restoration task(s) to be applied. (e.g., 'super-resolution', 'denoise+dehaze')

FaceRestore: True              # enable / disable face restoration module in 4KAgent
Brightening: False             # enable / disable brightening task in 4KAgent
OldPhotoRestoration: False     # enable / disable old photo restoration module in 4KAgent

RestorePerference: Perception  # [Fidelity, Perception]

Fast4K: True                   # enable / disable Fast4K mode in 4KAgent
Fast4kSideThres: 1024          # size threshold when enable Fast4K mode

User_Define: False             # Indicates whether to enable a user-specified plan instead of relying on VLM perception
User_Define_Plan: None         # Specifies the designated plan, if explicitly provided. (e.g., ["denoising", "super-resolution", "super-resolution"])

with_rollback: True            # Whether to trigger rollback in the system
```

All these settings have default value in 4KAgent (L128 ~ L171 in `the4kagent_pipeline.py`).


We evaluate 4KAgent in different tasks under different profiles, we conclude their details and naming convention here:

| Profile Nickname | Perception Agent        | Upscale to 4K | Scale Factor | Restore Option       | Face Restore | Brightening | Restore Preference |
| --------------- | --------------------- | ------------ | ----------- | ------------------- | ----------- | ---------- | ----------------- |
| Gen4K-P          | DepictQA         | True          | None         | None                 | True         | False       | Perception         |
| Gen4K-F          | DepictQA        | True          | None         | None                 | True         | False       | Fidelity           |
| Aer4K-P          | Llama-3.2-Vision  | True          | None         | None                 | False        | False       | Perception         |
| Aer4K-F          | Llama-3.2-Vision  | True          | None         | None                 | False        | False       | Fidelity           |
| ExpSR-s4-P       | Llama-3.2-Vision  | False         | 4            | super-resolution     | False        | False       | Perception         |
| ExpSR-s4-F       | Llama-3.2-Vision  | False         | 4            | super-resolution     | False        | False       | Fidelity           |
| ExpSR-s2-F       | Llama-3.2-Vision  | False         | 2            | super-resolution     | False        | False       | Fidelity           |
| ExpSR-s8-F       | Llama-3.2-Vision  | False         | 8            | super-resolution     | False        | False       | Fidelity           |
| GenSR-s4-P       | DepictQA        | False         | 4            | None                 | False        | False       | Perception         |
| AerSR-s4-P       | Llama-3.2-Vision  | False         | 4            | None     | False        | False       | Perception         |
| AerSR-s4-F       | Llama-3.2-Vision  | False         | 4            | None     | False        | False       | Fidelity           |
| GenMIR-P         | DepictQA        | False         | 4            | None                 | False        | True        | Perception         |
| ExpSRFR-s4-P     | Llama-3.2-Vision  | False         | 4            | super-resolution     | True         | False       | Perception         |
| GenSRFR-s4-P     | DepictQA        | False         | 4            | None                 | True         | False       | Perception         |

**Profile naming convention:** We combine _restoration type_, _restoration task_, and _restoration preference_ to construct the profile name. For example, **Gen** indicates a <u>General</u> image, **4K** indicates "Upscale to <u>4K</u>" on, and **P** indicates to restore the image with high <u>Perceptual</u> quality. **Aer** indicates <u>Aerial</u> image, **Exp** corresponds to <u>Explicit</u> setting, indicating that the profile has explicitly set the restoration task (e.g., **SR**, which indicates <u>Super-Resolution</u>). **MIR** indicates <u>Multiple-Degradation Image Restoration</u>. **FR** indicates <u>Face Restoration</u>. **s4** indicates to upscale the image by a <u>s</u>cale factor of <u>4</u>.