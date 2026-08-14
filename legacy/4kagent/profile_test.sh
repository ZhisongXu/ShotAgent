CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/classicsr \
  --output_dir ./outputs/4KAgent_test/classicsr \
  --profile_name ExpSR_s4_F \
  --tool_run_gpu_id 2

CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/realworldsr \
  --output_dir ./outputs/4KAgent_test/realworldsr \
  --profile_name ExpSR_s4_P \     # or GenSR_s4_P
  --tool_run_gpu_id 2

CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/mir \
  --output_dir ./outputs/4KAgent_test/mir \
  --profile_name GenMIR_P \
  --tool_run_gpu_id 2

CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/face \
  --output_dir ./outputs/4KAgent_test/face \
  --profile_name GenSRFR_s4_P \
  --tool_run_gpu_id 2

CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/opr \
  --output_dir ./outputs/4KAgent_test/opr \
  --profile_name OldP4K_P \
  --tool_run_gpu_id 2

CUDA_VISIBLE_DEVICES=1 python infer_4kagent.py \
  --input_dir ./assets/profile_test_example/4ksr \
  --output_dir ./outputs/4KAgent_test/4ksr \
  --profile_name FastGen4K_P \
  --tool_run_gpu_id 2