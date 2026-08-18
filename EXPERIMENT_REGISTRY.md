# Experiment Registry

Generated: `2026-08-17T19:16:35.085069+00:00`

This file is generated from existing outputs by `scripts/update_experiment_registry.py`.

## Pair leaderboard

| Pair | State | Common rounds | KFAC Last10 | Uniform Last10 | Gap | Valid converged |
|---|---:|---:|---:|---:|---:|---:|
| cifar10_a0.1_seed0_resnet18_gn_e16_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_r50_20260817_170812 | running/completed | 20 | 56.703% | 55.819% | +0.884 pp | no |
| cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs32_lr0.001_lb0.0_20260817_034759 | partial_stopped/completed | 160 | 75.012% | 75.869% | -0.857 pp | no |
| cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs32_lr0.001_lb0.0_20260817_082813 | completed/completed | 200 | 78.496% | 78.623% | -0.127 pp | no |
| cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_gain2p5_r50_20260817_143302 | completed | 0 | - | - | - | no |
| cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_r50_20260817_104700 | completed/completed | 50 | 67.253% | 67.462% | -0.209 pp | no |

## Method runs

### cifar10_a0.1_seed0_resnet18_gn_e16_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_r50_20260817_170812

- `expert_local_kfac_whiten_layer_projection`: running, rounds 20/50, Last10 56.703%, best 60.020% at round 20, final-window converged=False (best improvement 8.44999999999999).
- `expert_uniform_all_valid_denominator`: completed, rounds 50/50, Last10 66.930%, best 68.280% at round 45, final-window converged=False (best improvement 0.6299999999999972).

### cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs32_lr0.001_lb0.0_20260817_034759

- `expert_local_kfac_whiten_layer_projection`: partial_stopped, rounds 160/200, Last10 75.012%, best 75.500% at round 157, final-window converged=False (best improvement 0.5000000000000004).
- `expert_uniform_all_valid_denominator`: completed, rounds 200/200, Last10 78.623%, best 79.140% at round 192, final-window converged=False (best improvement 0.6800000000000028).

### cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs32_lr0.001_lb0.0_20260817_082813

- `expert_local_kfac_whiten_layer_projection`: completed, rounds 200/200, Last10 78.496%, best 78.800% at round 200, final-window converged=False (best improvement 0.26000000000000467).
- `expert_uniform_all_valid_denominator`: completed, rounds 200/200, Last10 78.623%, best 79.140% at round 192, final-window converged=False (best improvement 0.6800000000000028).

### cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_gain2p5_r50_20260817_143302

- `expert_local_kfac_whiten_layer_projection`: completed, rounds 50/50, Last10 67.148%, best 67.900% at round 49, final-window converged=False (best improvement 0.29000000000000137).

### cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_r50_20260817_104700

- `expert_local_kfac_whiten_layer_projection`: completed, rounds 50/50, Last10 67.253%, best 68.520% at round 45, final-window converged=False (best improvement 0.990000000000002).
- `expert_uniform_all_valid_denominator`: completed, rounds 50/50, Last10 67.462%, best 68.220% at round 49, final-window converged=False (best improvement 0.45000000000000595).
