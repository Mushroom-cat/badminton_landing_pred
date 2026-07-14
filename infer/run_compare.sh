python compare_duida_pose_fit_predictions.py \
  --pose-model ../models/before.pt \
  --fit-model ../models/c_parameter_mlp_fall.pt \
  --pose-mode before \
  --pose-dir '../datasets/2026-7-10test/4-poseball' \
  --fit-dir '../datasets/2026-7-10test/3-fit' \
  --pose-skip-n 5
