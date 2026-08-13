tmux new -s moe
tmux attach -t moe

cd /home/cjq/Project/fl_moe
conda activate fl_moe

CUDA_VISIBLE_DEVICES=2 python train.py


python plot_results.py \
  --csv-a outputs/cifar10/resnet18_gn/uniform/seed_0/20260715_092727_227646/metrics.csv \
  --label-a Uniform \
  --csv-b outputs/cifar10/resnet18_gn/activation_weighted/seed_0/20260715_092710_240500/metrics.csv \
  --label-b "Activation Weighted" \
  --window 5 \
  --output pictures/c10_r200_top1_dim1024.png