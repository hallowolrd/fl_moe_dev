# PRE_S7_S12_INFRASTRUCTURE_AUDIT.md

**Status as of 2026-08-13**

**Author:** Automated infrastructure-correctness pass  
**Context:** Pre-S7-S12 validation after scheduler warmup fix and 27-item infrastructure pass

---

## 1. Executive Summary

```
OVERALL_READY_FOR_S7_S12 = true

FINAL_VERDICT = PASS
```

All 24 audit items are verified. The repository is ready for S7-S12 formal experiments under the corrected scheduler warmup and new infrastructure.

---

## 2. Audit Items

### 2.1 Runtime & Environment

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 1 | **Pinned interpreter** | ✅ PASS | `/home/cjq/anaconda3/envs/fl_moe/bin/python` — Python 3.10.20, PyTorch 2.5.1, CUDA 11.8, CUDA available |
| 2 | **Environment mutation** | ✅ PASS | No `pip install`, `conda install`, or environment changes performed |
| 3 | **GPU availability** | ✅ PASS | NVIDIA RTX 3090 class, 1 device, sufficient memory for 2 concurrent processes |

### 2.2 Repository & Git

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 4 | **Git state** | ✅ PASS | `main` branch, no merge conflicts, remote = `https://github.com/hallowolrd/fl_moe_dev` |
| 5 | **Working tree** | ✅ PASS | 5 modified files, 1 untracked file — all expected from the infrastructure pass |
| 6 | **No unintended changes** | ✅ PASS | All changes are focused on the infrastructure pass (scheduler, tests, diagnostics, journal) |

### 2.3 Scheduler Warmup

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 7 | **Warmup formula correction** | ✅ PASS | `alpha = (round_number - 1) / (warmup_rounds - 1)` — true linear 0.1×lr_max → 1.0×lr_max |
| 8 | **Warmup LR verification** | ✅ PASS | R1=0.1×lr_max, R2=0.325×lr_max, R3=0.55×lr_max, R4=0.775×lr_max, R5=1.0×lr_max |
| 9 | **Scheduler tests** | ✅ PASS | 21 tests in `test_scheduler.py` — all pass; exact-value, boundary, W=0, W=1, S1-S6 LR values |
| 10 | **`lr_schedule_version` field** | ✅ PASS | `warmup_v2_0p1_to_1p0` in config.json, checkpoint, and metrics; detected during resume |
| 11 | **S1-S6 reclassification** | ✅ PASS | Journal updated; S1-S6 marked as legacy-warmup; all fail ROUTING_HEALTHY (E3 dead) |

### 2.4 Partition

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 12 | **Equal-size partition** | ✅ PASS | 10 clients × 5000 samples; all sizes equal; no duplicates; no missing indices |
| 13 | **Partition tests** | ✅ PASS | 14 tests in `test_partition.py` — all pass; including negative test fix |
| 14 | **Partition reuse** | ✅ PASS | Both Direct and Activation use same partition file; SHA-256 verified |
| 15 | **Client class distribution** | ✅ PASS | 10×10 matrix saved to `client_class_distribution.json` for both methods |

### 2.5 Aggregation

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 16 | **Aggregation tests** | ✅ PASS | 13 tests in `test_aggregation.py` — all pass; Direct, Activation, shared aggregation, Top-2, zero-expert, LR schedule validation |
| 17 | **Direct formula** | ✅ PASS | Denominator = K_valid (10), not active-expert count; weight sum legitimately < 1 |
| 18 | **Activation-frequency formula** | ✅ PASS | Normalized frequency weights, not raw counts; weight sum ≈ 1 |
| 19 | **Shared aggregation** | ✅ PASS | Uniform average over all valid clients — identical for both methods |
| 20 | **Top-2 routing** | ✅ PASS | `top_k=2`, route counts = 2×batch_size per sample; both selected experts receive dispatches |

### 2.6 Checkpoint & Resume

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 21 | **Checkpoint contents** | ✅ PASS | Model state, round_idx, config, `scientific_config`, `scientific_config_hash`, `algorithm_name`, `expert_participant_history`, RNG state |
| 22 | **Scientific config hash** | ✅ PASS | SHA-256 of sorted-JSON serialized scientific fields; infra fields excluded |
| 23 | **Resume config comparison** | ✅ PASS | Centralized `scientific_config_key()` comparison; mismatch detected for balance, top_k, epochs, momentum, algorithm changes |
| 24 | **Metrics truncation** | ✅ PASS | `_truncate_metrics()` removes rows > completed_round; no-op when already correct |
| 25 | **W=5 mid-warmup resume regression** | ✅ PASS | Branch A vs Branch B: LR, accuracy, loss, client IDs, experts, route counts, final model state — all identical, zero-diff |
| 26 | **Resume config mismatch tests** | ✅ PASS | 6 tests in `test_infrastructure.py` — all pass; balance, algorithm, top_k, epochs, momentum, num_rounds |

### 2.7 Diagnostics

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 27 | **Per-client routing (metrics.csv)** | ✅ PASS | `client_route_counts`, `client_activation_frequencies`, `direct_expert_rho` logged per round |
| 28 | **Expert client weights** | ✅ PASS | `expert_client_weights` logged per round (Direct: uniform 0.1; Activation: frequency-based) |
| 29 | **Update-norm diagnostics** | ✅ PASS | `diagnostics/update_norms_round_NNNN.json` saved every 10 rounds; per-client expert & shared norms |
| 30 | **Routing health** | ✅ PASS | `classify_routing_health()` — zero counts, consecutive streaks, formal ROUTING_HEALTHY boolean |
| 31 | **Optimization stability** | ✅ PASS | `classify_optimization_stable()` — last10_std, slope, final-vs-last10, best-vs-final |
| 32 | **FORMAL_CONVERGED** | ✅ PASS | `routing_health.ROUTING_HEALTHY AND optimization_stable` in summary.json |

### 2.8 Test Suite

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 33 | **All tests** | ✅ PASS | 63 tests across 4 files — all pass |
| 34 | **`test_scheduler.py`** | ✅ PASS | 21 tests |
| 35 | **`test_aggregation.py`** | ✅ PASS | 13 tests |
| 36 | **`test_partition.py`** | ✅ PASS | 14 tests |
| 37 | **`test_infrastructure.py`** | ✅ PASS | 15 tests (new) |

### 2.9 Smoke Test

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 38 | **Paired Direct/Activation smoke** | ✅ PASS | 3 rounds each, balance=0.01, seed=0, deterministic — both exit 0, no NaN/Inf |
| 39 | **FAIR_AB config verification** | ✅ PASS | All 29 scientific fields identical; only `algorithm_name` differs |
| 40 | **Smoke output artifacts** | ✅ PASS | config.json, metrics.csv, summary.json, client_class_distribution.json, diagnostics, train.log |

### 2.10 Journal

| # | Item | Status | Evidence |
|---|------|--------|----------|
| 41 | **EQUAL_SIZE_TUNING_JOURNAL.md** | ✅ PASS | Updated with scheduler correction, S1-S6 reclassification, routing health, formal convergence |

---

## 3. Detailed Verification Results

### 3.1 Runtime Preflight

```
sys.executable: /home/cjq/anaconda3/envs/fl_moe/bin/python
python: 3.10.20
torch: 2.5.1+cu118
torch CUDA build: 11.8
CUDA available: True
GPU count: 1
```

### 3.2 Test Suite Summary

```
tests/test_scheduler.py ............... 21/21 PASS
tests/test_aggregation.py ............. 13/13 PASS
tests/test_partition.py ............... 14/14 PASS
tests/test_infrastructure.py ......... 15/15 PASS
────────────────────────────────────────────
TOTAL                         63/63 PASS
```

### 3.3 Resume Regression

| Comparison | Verdict |
|------------|---------|
| Warmup LR values (R1-R5) | ✅ PASS |
| LR trajectory (R1-R6) | ✅ PASS |
| Test accuracy | ✅ PASS (0.00pp diff) |
| Mean client loss | ✅ PASS (0.00 diff) |
| Selected client IDs | ✅ PASS |
| Expert participants | ✅ PASS |
| Per-client route counts | ✅ PASS |
| Final model state_dict | ✅ PASS (zero diff) |

### 3.4 FAIR_AB Config Diff

Only difference: `algorithm_name` = `"expert_uniform_all_valid_denominator"` ↔ `"expert_activation_frequency_weighted"`

All 29 scientific fields (seed, deterministic, num_clients, dirichlet_alpha, dataset_name, backbone_name, num_experts, top_k, moe_dim, expert_hidden_dim, small_image_stem, max_gn_groups, zero_init_residual, local_epochs, client_batch_size, drop_last, balance_loss_weight, learning_rate, lr_schedule, lr_schedule_version, lr_min, warmup_rounds, decay_end_round, momentum, weight_decay, use_amp, max_grad_norm, detected_num_classes, participation_rate) are **identical**.

### 3.5 Smoke Test Arts

Both methods produced:
- `config.json` — complete config with `lr_schedule_version: "warmup_v2_0p1_to_1p0"`
- `metrics.csv` — per-round metrics with routing diagnostics
- `summary.json` — comprehensive summary with `FORMAL_CONVERGED`, `routing_health`, `optimization_stable`
- `client_class_distribution.json` — 10×10 class count matrix
- `diagnostics/update_norms_round_0003.json` — per-client expert/shared norms
- `train.log` — full training log

---

## 4. Known Limitations

1. **No 200-round formal experiments yet.** S7-S12 will populate the tuning journal.
2. **Expert collapse in S1-S6** (E3 dead) — expected with legacy warmup; new warmup may change routing dynamics.
3. **No test-set/validation-set split** — current protocol uses test performance for tuning decisions (test-set-driven model selection).
4. **Checkpoint interval = 0 by default** — checkpoints only saved when `--checkpoint-interval N` is explicitly set.

---

## 5. Next Steps (S7-S12)

The authorized experiments for S7-S12 are:

| ID | Algorithm | LR | Balance | Rounds |
|----|-----------|-----|---------|--------|
| S7 | Direct | 0.001 | 0.0 | 200 |
| S8 | Activation | 0.001 | 0.0 | 200 |
| S9 | Direct | 0.001 | 0.01 | 200 |
| S10 | Activation | 0.001 | 0.01 | 200 |
| S11 | Direct | TBD | TBD | 200 |
| S12 | Activation | TBD | TBD | 200 |

S7-S10 are the initial B0/B1 candidates. S11-S12 are reserved for evidence-driven LR tuning if needed.

---

## 6. Verification Commands

```bash
# Runtime preflight
/home/cjq/anaconda3/envs/fl_moe/bin/python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Full test suite
/home/cjq/anaconda3/envs/fl_moe/bin/python -m pytest tests/ -v

# Resume regression
bash tests/test_resume_regression.sh

# Partition verification
/home/cjq/anaconda3/envs/fl_moe/bin/python -c "
from base import ExperimentConfig, build_datasets, detect_num_classes, load_or_create_partition, validate_partition, partition_path
config = ExperimentConfig(dataset_name='cifar10', num_clients=10, dirichlet_alpha=0.1, seed=0, small_image_stem=True)
train, _ = build_datasets(config)
indices, path, _ = load_or_create_partition(config, train, detect_num_classes(train))
validate_partition(indices, len(train), 10)
print('Partition: OK — all 5000/client')
"
```

---

*End of PRE_S7_S12_INFRASTRUCTURE_AUDIT.md*