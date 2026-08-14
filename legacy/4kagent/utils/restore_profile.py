# General_SuperResolution_Fidelity_ToolName = ["hat_psnr", "xrestormer", "pisasr_psnr", "hma"]
General_SuperResolution_Fidelity_ToolName = ["hat_psnr", "xrestormer", "swinfir", "hma", "drct"]
General_SuperResolution_Perception_ToolName = ["hat_gan", "swinir_gan", "diffbir", "osediff", "pisasr"]

# 2x super resolution
General_SuperResolution_2x_Fidelity_ToolName = ["hat_psnr_2x", "swinfir_2x", "hma_2x"]
General_SuperResolution_2x_Perception_ToolName = ["swinir_2x_gan", "diffbir_2x", "osediff_2x", "pisasr_2x"]

# directly 16x super resolution
General_SuperResolution_16x_Perception_ToolName = ["osediff_16x", "pisasr_16x"]

# Baseline Toolbox
General_Baseline_Brightening_ToolName = ["constant_shift", "gamma_correction", "histogram_equalization"]
General_Baseline_Defocus_Deblurring_ToolName = ["drbnet", "ifan", "restormer"]
General_Baseline_Dehazing_ToolName = ["maxim", "xrestormer", "ridcp", "dehazeformer"]
General_Baseline_Denoising_ToolName = ["swinir_15", "swinir_50", "maxim", "mprnet", "restormer", "xrestormer"]
General_Baseline_Deraining_ToolName = ["maxim", "mprnet", "restormer", "xrestormer"]
General_Baseline_Jpeg_Compression_Artifact_Removal_ToolName = ["swinir_40", "fbcnn_blind", "fbcnn_5", "fbcnn_90"]
General_Baseline_Motion_Deblurring_ToolName = ["maxim", "mprnet", "restormer", "xrestormer"]
General_Baseline_SuperResolution_ToolName = ["diffbir", "xrestormer", "hat_psnr", "swinir_psnr", "swinir_gan"]