# Equal-Size FL-MoE Tuning Journal

All experiments recorded below use the **new equal-size balanced label-Dirichlet protocol** (FedDyn-style, 5000 samples/client).

Legacy unequal-size experiments are historical/diagnostic only and are not part of this formal tuning history.

---

## S1-S6: LR × Decay-End-Round Sweep (100 rounds)

### Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | CIFAR-10 |
| Partition | feddyn_balanced_dirichlet (10 clients, α=0.1, 5000/client) |
| Model | ResNet18-GN → 4 experts (Top-2) |
| Rounds | 100 |
| LR schedule | Cosine (warmup 5 rounds, lr_min=5e-5) |
| Balance loss weight | 0.01 |
| Local epochs | 1 |
| Batch size | 64 |
| Seed | 0 |
| Deterministic | True |

### Scheduler candidates

| ID | LR max | Decay end round |
|----|--------|-----------------|
| S1 | 0.0015 | 70 |
| S2 | 0.0015 | 80 |
| S3 | 0.0020 | 70 |
| S4 | 0.0020 | 80 |
| S5 | 0.0025 | 70 |
| S6 | 0.0025 | 80 |

### Results

| ID | Direct final | Direct best | Direct last10 | Activation final | Activation best | Activation last10 | Gap (pp) | Converged |
|----|-------------|-------------|---------------|-----------------|-----------------|-------------------|----------|-----------|
| S1 | 54.27% | 54.41% | 54.21±0.15% | 53.98% | 54.39% | 54.13±0.17% | -0.29 | ✓ |
| S2 | 56.12% | 56.30% | 56.04±0.19% | 55.57% | 56.16% | 55.70±0.12% | -0.55 | ✓ |
| S3 | 55.68% | 55.99% | 55.66±0.20% | 56.09% | 56.34% | 56.09±0.14% | +0.41 | ✓ |
| S4 | 57.85% | 58.15% | 57.95±0.13% | 59.12% | 59.13% | 58.94±0.12% | **+1.27** | ✓ |
| S5 | 57.70% | 57.87% | 57.67±0.12% | 57.76% | 57.99% | 57.76±0.14% | +0.06 | ✓ |
| S6 | **59.34%** | **59.51%** | **59.33±0.13%** | **59.80%** | **60.06%** | **59.88±0.10%** | +0.46 | ✓ |

### Target Assessment

**Level C — target not reached**

| Condition | Requirement | Best achieved | Met? |
|-----------|------------|---------------|------|
| Direct accuracy | [60%, 65%] | 59.34% (S6) | ✗ (0.66pp below) |
| Activation accuracy | [64%, 68%] | 59.80% (S6) | ✗ (4.20pp below) |
| Gap | ≥ 3.00pp | +1.27pp (S4) | ✗ (1.73pp below) |
| Both converged | last10_std ≤ 2pp, | all ≤ 0.20pp | ✓ |
| FAIR_AB | true | true | ✓ |

### Key Findings

1. **Higher LR improves accuracy.** S6 (lr=0.0025, d80) achieves 59.34% Direct and 59.80% Activation, significantly higher than S1 (lr=0.0015) at 54.27%/53.98%.

2. **Longer decay helps for higher LR.** S6 (d80) outperforms S5 (d70) by +1.64pp Direct and +2.04pp Activation, suggesting the higher LR benefits from more training time before decay.

3. **Activation gap is small.** The maximum Activation advantage is only +1.27pp (S4). S1, S2, and S5 show essentially no gap. S4 shows the most consistent Activation advantage.

4. **Expert 3 collapse is persistent.** Direct always has 0 participants for expert 3. Activation sometimes recovers some activity (S4: 8 participants at R100, S3: 1-2), but this doesn't translate to a large accuracy gap.

5. **All experiments converged well.** Last 10-round standard deviations are ≤ 0.20pp, indicating stable convergence.

6. **S6 is the best overall candidate** but falls short of Level A targets.

### Discussion

The Activation-frequency weighting provides at most a 1.27pp advantage over Direct uniform weighting at 100 rounds. This is far below the 3.00pp target. The small gap is likely because:

- The equal-size partition means all clients have identical data volumes, so uniform weighting (Direct) is already quite fair
- Activation-frequency weighting primarily helps when clients have different data quantities, which is controlled here
- Expert 3 collapse limits both methods, especially Direct which never recovers expert 3

### Next Steps

The evidence suggests that:
1. **Accuracy is still below 60%** even at lr=0.0025. Further LR increase (0.0030+) might help.
2. **The gap is fundamentally small** regardless of LR/decay settings.
3. **Expert collapse** is a limiting factor.

Options for the next phase:
- **A: Increase LR further** — try lr=0.0030, 0.0035 with decay_end_round=80
- **B: Increase rounds** — try 200 rounds with S6 config to see if further convergence helps
- **C: Address expert collapse** — investigate router initialization or balance loss
- **D: Report current results** as the best reproducible FAIR_AB comparison under the protocol