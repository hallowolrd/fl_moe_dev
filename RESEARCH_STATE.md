# FL-MoE Phase-1 Research State

Updated: 2026-08-17 11:23 UTC

## Locked objective

Maximize seed-0 `Gap_Last10` between strict-reference Local-KFAC and the
unchanged all-valid-denominator Uniform baseline under the locked CIFAR-10,
alpha 0.1, Top-2, ResNet18+MoE protocol. A valid final pair requires both
methods to pass the 10-round / 0.1 percentage-point best-improvement plateau
gate, and Uniform must reach at least 60% test accuracy.

## Current best status

- No valid converged pair exists yet.
- Historical no-replay KFAC stopped at round 160 and had a matched round-160
  `Gap_Last10 = -0.857 pp`.
- The running standard-augmentation strict-replay pair had reached KFAC round
  62 and Uniform round 101 at the latest audit. Its matched round-50 gap was
  `+0.678 pp`; it is not converged and is not duplicated.
- The current-HEAD high-accuracy no-augmentation screen is running. Uniform
  reached 61.10% at round 20; its round-11--20 mean is 57.544%. It first
  crossed the 60% validity floor at round 18 but remains unconverged.

## Current hypothesis

The strongest known high-accuracy neighborhood should be validated under the
current no-augmentation and strict occurrence-level replay semantics:

- 8 experts, Top-2
- client batch size 64
- 2 local epochs
- learning rate 0.002
- MoE dimension 512
- expert hidden dimension 512
- balance-loss weight 0.003

This joint candidate is intentionally treated as a historically motivated
starting point, not as proof that each component is individually optimal.
Existing diagnostics also show that the projection is nearly inactive after
early rounds, so route-weighting attribution and correction outliers remain
important mechanism questions.

## Running experiments

- `outputs/pair_runs/cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs32_lr0.001_lb0.0_20260817_082813`
  - standard augmentation
  - KFAC on physical GPU 0
  - Uniform on physical GPU 1
  - launched from recorded commit `529cd2e` with a dirty KFAC source
- `outputs/pair_runs/cifar10_a0.1_seed0_resnet18_gn_e8_top2_bs64_le2_lr0.002_lb0.003_m512_h512_augnone_r50_20260817_104700`
  - current commit semantics, explicit no augmentation and strict replay
  - 50-round high-accuracy-neighborhood screen
  - sequential Uniform then KFAC on physical GPU 5
  - KFAC smoke passed at about 1.45 GiB observed peak VRAM
  - Uniform reached round 20 with 61.10% test accuracy at the latest update
  - round-11--20 mean-route effective utilization was 3.47/8; expert 5 was dead

## Rejected or non-final evidence

- `20260817_034759`: legacy no-replay KFAC, incomplete and source-dirty.
- `learning_rate=0.01`: explicitly recorded as failed; do not retry.
- Large positive gaps from an unconverged or sub-60% Uniform run are invalid.

## Next action

1. Monitor the current high-accuracy screen through Uniform and KFAC round 50.
2. Update the registry at milestones and analyze rolling gap, convergence, routing,
   KFAC coverage/projection/correction, fallback reasons, and replay consistency.
3. Use the round-50 evidence to select the next single high-information Phase-1
   ablation or promote the candidate to 100 rounds.
