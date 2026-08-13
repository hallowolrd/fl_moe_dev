---

name: fl-moe-equal-size-top2-target-tuner
version: 2026-08-13-equal-size-top2-v1
summary: >
Audit, validate, tune, and finalize the user's CIFAR-10 federated Sparse-MoE
system in /home/cjq/Project/fl_moe using an equal-size FedDyn-style balanced
label-Dirichlet partition. Compare expert_uniform_all_valid_denominator
against expert_activation_frequency_weighted under a strict FAIR_AB protocol.
Keep the current base.py model architecture fixed, use 4 experts and formal
Top-2 routing, preserve the pinned runtime environment, execute long runs
disconnect-safely, and search only a tightly controlled shared training space
for a reproducible configuration targeting Direct 60-65%, Activation 64-68%,
and Activation-minus-Direct >= 3.00 percentage points.
------------------------------------------------------

# FL-MoE Equal-Size Top-2 Target Tuner

## 1. Mission and Formal Scientific Target

Work inside the user's existing local repository:

```text
LOCAL_REPOSITORY = /home/cjq/Project/fl_moe
```

The corresponding remote repository is:

```text
REMOTE_REPOSITORY = https://github.com/hallowolrd/fl_moe_dev
```

The local directory name and remote repository name are intentionally different. Do not rename the local repository merely to match the remote.

The task is to build, validate, tune, and finalize a fair Federated Learning + Sparse Mixture-of-Experts experiment comparing exactly two expert aggregation strategies:

```text
A = expert_uniform_all_valid_denominator
B = expert_activation_frequency_weighted
```

The final scientific comparison must isolate:

```text
expert_aggregation
```

as the only scientific difference between A and B.

### 1.1 Formal experiment protocol

```text
dataset                  = CIFAR-10
num_clients              = 10
dirichlet_alpha          = 0.1
participation_rate       = 1.0

partition                = equal-size FedDyn-style balanced label-Dirichlet

model                    = current base.py model architecture
backbone                 = current ResNet18-GN implementation
num_experts              = 4
formal_top_k             = 2

local_objective          = standard CE + optional common balance loss

shared_aggregation       = uniform over all valid client updates

formal_rounds            = 200
seed                     = 0
deterministic            = true
```

### 1.2 Exact target

The requested Level A target remains:

```text
Direct final accuracy      ∈ [60%, 65%]
Activation final accuracy  ∈ [64%, 68%]

Activation - Direct        >= 3.00 percentage points

both methods converged
FAIR_AB = true
```

Define:

```text
gap_pp =
    final_accuracy(expert_activation_frequency_weighted)
  - final_accuracy(expert_uniform_all_valid_denominator)
```

Exact success requires all conditions simultaneously.

Reaching the two accuracy ranges without reaching:

```text
gap_pp >= 3.00
```

is not exact success.

These are optimization targets, not permission to manipulate the experiment.

Never create the requested gap using:

* different partitions;
* different seeds;
* different training budgets;
* different learning rates;
* method-specific balance-loss values;
* different model architectures;
* different checkpoint selection;
* different evaluation procedures;
* intentional degradation of Direct;
* altered aggregation formulas.

If the exact target cannot be reached under a scientifically fair common configuration, report the nearest reproducible fair result.

---

# 2. Hard Runtime, Environment, and Repository Safety

## 2.1 Pinned runtime

The only authorized repository Python interpreter is:

```text
CONDA_ENV_NAME     = fl_moe
CONDA_ENV_PREFIX   = /home/cjq/anaconda3/envs/fl_moe
PYTHON_EXECUTABLE  = /home/cjq/anaconda3/envs/fl_moe/bin/python
```

Known verified runtime:

```text
Python             = 3.10.x
PyTorch            = 2.5.1
PyTorch CUDA build = 11.8
CUDA available     = true
GPU environment    = NVIDIA RTX 3090 class
```

Every repository Python command must explicitly use:

```bash
/home/cjq/anaconda3/envs/fl_moe/bin/python
```

Examples:

```bash
/home/cjq/anaconda3/envs/fl_moe/bin/python -m pytest ...

CUDA_VISIBLE_DEVICES=0 \
/home/cjq/anaconda3/envs/fl_moe/bin/python \
experiments/expert_uniform_all_valid_denominator.py ...
```

Do not rely on:

```text
python
python3
pytest
conda activate
PATH ordering
VS Code terminal inheritance
```

to select the runtime.

## 2.2 Mandatory runtime preflight

Before the first Python-dependent repository task in each new agent session, and before every new formal experiment batch, execute:

```bash
/home/cjq/anaconda3/envs/fl_moe/bin/python - <<'PY'
import sys
import torch

expected = "/home/cjq/anaconda3/envs/fl_moe/bin/python"

print("sys.executable:", sys.executable)
print("python:", sys.version)
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if sys.executable != expected:
    raise RuntimeError(
        f"Wrong Python interpreter: {sys.executable}; expected {expected}"
    )

if sys.version_info[:2] != (3, 10):
    raise RuntimeError(
        f"Wrong Python version: {sys.version_info[:3]}; expected 3.10.x"
    )

if torch.__version__.split("+")[0] != "2.5.1":
    raise RuntimeError(
        f"Unexpected PyTorch version: {torch.__version__}; expected 2.5.1"
    )

if torch.version.cuda != "11.8":
    raise RuntimeError(
        f"Unexpected PyTorch CUDA build: {torch.version.cuda}; expected 11.8"
    )

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in pinned fl_moe environment")
PY
```

If this fails:

```text
RUNTIME_INTERPRETER_OK = false
ENVIRONMENT_BLOCKER = true
```

Stop Python-dependent work and report the exact mismatch.

Do not autonomously switch interpreters or repair the environment.

## 2.3 Environment mutation prohibited

By default:

```text
ENVIRONMENT_MUTATION_ALLOWED = false
```

Without explicit user approval for the exact operation, do not execute:

```text
pip install
pip uninstall
pip upgrade

python -m pip install

conda install
conda remove
conda update
conda create
conda clean

mamba/micromamba package changes

apt/apt-get/yum/dnf package changes

CUDA installation/removal
cuDNN changes
NVIDIA driver changes
PyTorch replacement

.bashrc changes
.profile changes
persistent PATH changes
persistent LD_LIBRARY_PATH changes

virtual-environment creation/removal
package-cache deletion
dataset deletion
```

A failed test is not authorization to repair the machine.

Read-only package inspection is allowed through the pinned interpreter.

## 2.4 Repository preflight

Before modifying repository files:

```bash
cd /home/cjq/Project/fl_moe

pwd
git status --short
git rev-parse --show-toplevel
git rev-parse HEAD
git remote -v
```

Expected local root:

```text
/home/cjq/Project/fl_moe
```

Expected remote repository:

```text
https://github.com/hallowolrd/fl_moe_dev
```

If the remote differs, report it.

Do not silently rewrite Git remotes.

Never:

```text
git reset --hard
git clean -fd
force checkout over user changes
discard pre-existing user modifications
```

Do not commit or push unless the user explicitly requests it.

Prefer small, reviewable repository changes.

---

# 3. Canonical Equal-Size Federated Partition

This section supersedes all legacy unequal-size Dirichlet experiments.

The formal partition is:

```text
PARTITION_METHOD =
FedDyn-style balanced label-Dirichlet

NUM_CLIENTS = 10
DIRICHLET_ALPHA = 0.1
```

## 3.1 Hard partition requirements

For CIFAR-10 training data:

```text
total training samples = 50,000
num clients            = 10
```

therefore the intended normal formal setting is:

```text
samples per client = 5,000
```

Every client must have exactly the same number of training samples.

Formally:

```text
n_0 = n_1 = ... = n_9
```

The partition must satisfy:

```text
all client sample counts equal
all intended training samples assigned exactly once
no duplicated training index
no omitted training index
no cross-client duplication
```

Heterogeneity must arise from:

```text
label/class composition
```

not from client data volume.

The formal experiment therefore studies:

```text
equal client quantity
+
heterogeneous label distribution
```

## 3.2 Partition reuse

All methods in a FAIR_AB comparison must reuse the exact same saved partition.

Never independently regenerate the Direct and Activation partitions.

Every serious run must record:

```text
partition_path
partition_method
partition_sha256
num_clients
alpha
client_sample_counts
```

A class-count matrix should also be saved or logged:

```text
client × class count matrix
```

## 3.3 Mandatory partition validation

Before formal training verify:

```text
NUM_CLIENTS == 10

max(client_sample_counts)
-
min(client_sample_counts)
== 0

len(all_assigned_indices)
== intended_training_set_size

len(set(all_assigned_indices))
== intended_training_set_size
```

Also verify the union of all client indices equals the intended training subset.

For the standard full CIFAR-10 training protocol:

```text
expected client_sample_counts =
[5000, 5000, 5000, 5000, 5000,
 5000, 5000, 5000, 5000, 5000]
```

If validation fails:

```text
PARTITION_VALID = false
```

Do not launch formal experiments.

## 3.4 No silent data-protocol changes

Do not introduce:

* proxy holdout;
* validation holdout;
* discarded training samples;
* duplicated samples;
* oversampling;
* client-specific sample budgets;

unless the user explicitly approves a new common data protocol.

---

# 4. Canonical Model and Local Training

## 4.1 Formal model architecture

The formal model is the current model implemented in `base.py`.

Do not replace it with legacy Residual Feature Experts or another architecture.

The expected current architecture is conceptually:

```text
CIFAR-10 image
    ↓
ResNet18-GN backbone
    ↓
512-dimensional feature representation
    ↓
current feature normalization / LayerNorm path
    ↓
sample-level router
    ↓
4 classification experts
```

Each classification expert is currently of the form:

```text
512
→ 1024
→ activation
→ num_classes
```

The exact executable `base.py` implementation is the source of truth for minor implementation details.

Before training, Codex must audit the actual current code.

If the current implementation materially differs from this skill, do not silently rewrite it.

Report the mismatch and ask the user before altering scientific semantics.

## 4.2 Architecture tuning is disabled

Formal mainline:

```text
MODEL_ARCHITECTURE_TUNING_ALLOWED = false
```

Do not autonomously try:

```text
8 experts
Top-1 formal training
Residual Feature Experts
shared classifier variants
different backbones
different expert hidden dimensions
different MoE placement
BN instead of GN
larger/smaller expert networks
```

unless explicitly authorized by the user.

## 4.3 Formal routing

Formal experiments use:

```text
top_k = 2
num_experts = 4
```

These are protocol invariants.

The default value inside `base.py` does not need to be changed if the framework intentionally supports other values.

Formal commands must explicitly set:

```text
--top-k 2
```

and, where appropriate:

```text
--num-experts 4
```

Never assume a `base.py` default equals the formal protocol.

## 4.4 Local objective

The formal local classification objective is standard sample-mean cross entropy on the final MoE logits:

```text
classification_loss =
CE(final_moe_logits, targets)
```

The total local objective is:

```text
total_loss =
classification_loss
+
balance_loss_weight * balance_loss
```

No expert-equal classification loss is part of the formal training objective unless the user explicitly changes the protocol later.

## 4.5 Allowed balance-loss values

The initial shared tuning set is strictly:

```text
balance_loss_weight ∈ {0.0, 0.01}
```

The same value must be applied to Direct and Activation within a candidate pair.

For example:

```text
Candidate B0:
    Direct      balance=0.0
    Activation  balance=0.0

Candidate B1:
    Direct      balance=0.01
    Activation  balance=0.01
```

Never compare:

```text
Direct balance=0
vs
Activation balance=0.01
```

as evidence of aggregation superiority.

---

# 5. Canonical Shared / Non-Expert Aggregation

All non-expert parameters use the current common uniform aggregation.

Canonical rule:

```text
SHARED_AGGREGATION =
uniform average over all valid client updates
```

For valid client updates `k = 1 ... K_valid`:

```text
Delta_shared_global =
(1 / K_valid) * sum_k Delta_shared_k
```

This aggregation is identical for Direct and Activation.

Formal protocol:

```text
SHARED_AGGREGATION_TUNING_ALLOWED = false
```

Do not autonomously replace it with:

* sample-count FedAvg;
* activation weighting;
* expert-dependent shared weighting;
* meta weighting;
* Fisher/KFAC weighting;
* any method-specific shared aggregation.

Only expert aggregation is being compared.

---

# 6. Canonical Direct Expert Aggregation

Formal Direct method:

```text
expert_uniform_all_valid_denominator
```

Let:

```text
Delta[k,e]
```

be client `k`'s uploaded parameter delta for expert `e`.

Let:

```text
a[k,e]
```

be the actual route/dispatch count of expert `e` during local training.

Define the active client set for expert `e`:

```text
A_e = { k | a[k,e] > 0 }
```

Let:

```text
K_valid = total number of valid client updates in the round
```

The canonical Direct aggregation rule is:

```text
Delta_global[e] =
    sum_{k in A_e} Delta[k,e]
    /
    K_valid
```

Equivalently:

```text
w[k,e] =
    1 / K_valid    if k in A_e
    0              otherwise
```

### Critical rule

Do NOT renormalize over active clients.

The denominator is:

```text
K_valid
```

NOT:

```text
|A_e|
```

Therefore:

```text
sum_k w[k,e]
=
|A_e| / K_valid
```

and this may legitimately be less than 1.

This is intentional algorithm behavior, not a normalization bug.

If:

```text
A_e = empty
```

preserve the current server expert parameters for expert `e`.

Do not reinterpret the method as active-client uniform aggregation.

---

# 7. Canonical Activation-Frequency Expert Aggregation

Formal Activation method:

```text
expert_activation_frequency_weighted
```

Let:

```text
a[k,e]
```

be the actual number of times expert `e` was selected/dispatched by client `k` during the local training interval that produced the uploaded expert delta.

For each client define total routing activity:

```text
T_k = sum_r a[k,r]
```

Define client-local activation frequency:

```text
f[k,e] =
a[k,e] / T_k
```

when:

```text
T_k > 0
```

For each expert normalize activation frequency across clients:

```text
w[k,e] =
f[k,e]
/
sum_j f[j,e]
```

for positive-frequency contributors.

Then:

```text
Delta_global[e] =
sum_k w[k,e] * Delta[k,e]
```

If:

```text
sum_j f[j,e] = 0
```

preserve the previous global expert parameters for expert `e`.

For an active expert:

```text
sum_k w[k,e] ≈ 1
```

subject to normal floating-point error.

## 7.1 Frequency is not raw count

Do not replace:

```text
f[k,e]
```

with raw:

```text
a[k,e]
```

without explicit user approval.

The formal method is activation-frequency weighting, not legacy activation-count weighting.

Do not substitute:

* client sample count;
* router probability sum;
* gate score;
* loss;
* sample-count weighting;
* another proxy.

## 7.2 Why equal-size partition matters

In the formal protocol, client sample counts are equal.

Under equal training budgets and sample-level Top-2 routing:

```text
T_k
```

is correspondingly controlled across clients.

Therefore activation frequency represents the client's relative expert usage rather than uncontrolled client data volume.

This is a core reason for using the new equal-size partition.

---

# 8. FAIR_AB — One-Variable Scientific Comparison

The final A/B experiment is valid only when the sole scientific difference is:

```text
expert_aggregation
```

The two conditions are:

```text
A = expert_uniform_all_valid_denominator
B = expert_activation_frequency_weighted
```

Everything else must be identical.

At minimum verify equality of:

```text
dataset
training subset
partition file
partition SHA256
num_clients
Dirichlet alpha
participation rate
seed
deterministic setting

model architecture
backbone
normalization
num_experts
top_k
router
expert architecture

local objective
balance_loss_weight

optimizer
learning rate
momentum/betas
weight decay
batch size
local epochs
round count

shared aggregation
checkpoint policy
evaluation procedure
final metric definition
```

Administrative differences are allowed:

```text
method name
run ID
timestamp
output directory
log filename
GPU process metadata
```

## 8.1 Mandatory final semantic config diff

Before accepting a final pair, compare normalized configs.

The only acceptable scientific difference is equivalent to:

```text
expert_aggregation:
    expert_uniform_all_valid_denominator
    ->
    expert_activation_frequency_weighted
```

If a second scientific field differs:

```text
FAIR_AB = false
STATUS = invalid_comparison
```

Do not use the accuracy gap as evidence.

Only when the comparison is clean may Codex record:

```text
FAIR_AB = true
```

---

# 9. Allowed Shared Tuning Space

The new equal-size protocol starts a new tuning history.

Legacy unequal-size hyperparameter conclusions are not assumed to transfer.

## 9.1 Clean baseline

Initial common training configuration:

```text
learning_rate = 0.001
formal_rounds = 200
seed = 0
deterministic = true
num_experts = 4
top_k = 2
```

Other common optimizer/training fields initially use the current `base.py` formal defaults unless this skill explicitly fixes them.

## 9.2 First tuning dimension

The first allowed shared candidates are only:

```text
balance_loss_weight = 0.0
balance_loss_weight = 0.01
```

Each candidate is a full Direct/Activation pair.

## 9.3 Learning-rate tuning

If neither balance candidate yields a satisfactory converged result, shared learning-rate tuning may be considered.

Before launching a new LR search stage:

1. analyze the completed equal-size results;
2. state the reason LR is suspected;
3. propose a small evidence-based shared LR candidate set;
4. obtain user approval if the candidate values are not already explicitly authorized.

Any LR candidate must be applied identically to both methods.

Do not tune Activation's LR independently.

Do not tune Direct's LR independently.

## 9.4 Forbidden autonomous tuning

Without explicit user approval, do not tune:

```text
top_k
num_experts
model architecture
expert hidden dimension
backbone architecture
normalization family
shared aggregation
data partition method
Direct denominator
Activation formula
residual scaling
Residual Feature Experts
shared classifier architecture
```

Do not import old F1-F6/G1-G6/decay-horizon settings as canonical values.

---

# 10. Mandatory Correctness and Regression Tests

Before serious training, validate the current repository.

## 10.1 Partition tests

Test:

```text
num_clients == 10

all client sizes equal

all intended training indices appear exactly once

no duplicate indices

no missing indices

saved partition can be reloaded deterministically
```

For standard CIFAR-10 full training data:

```text
client size == 5000
```

for every client.

## 10.2 Top-2 routing tests

For formal Top-2:

```text
top_k == 2
topk_indices.shape == [batch_size, 2]
```

For sample-level routing:

```text
sum(route_counts)
==
batch_size * 2
```

Verify both selected experts actually receive dispatches.

Search the code for hidden Top-1 assumptions such as:

```text
topk_indices[:, 0]
top_k == 1
shape == [B,1]
```

A base default of `top_k=1` is not itself a bug if formal commands explicitly set Top-2.

## 10.3 Standard CE test

Verify backward propagation through:

```text
CE(final_moe_logits, targets)
```

and confirm balance loss is added only once:

```text
CE
+
balance_loss_weight * balance_loss
```

## 10.4 Direct synthetic aggregation test

Example:

```text
K_valid = 3

expert e active on:
client0
client1

client2 inactive
```

Expected weights:

```text
[1/3, 1/3, 0]
```

Expected weight sum:

```text
2/3
```

The test must fail if the implementation incorrectly renormalizes to:

```text
[1/2, 1/2, 0]
```

## 10.5 Activation-frequency synthetic test

Example:

```text
client0:
    expert e count = 10
    total route count = 20
    frequency = 0.5

client1:
    expert e count = 30
    total route count = 100
    frequency = 0.3
```

Then:

```text
normalization denominator = 0.8
```

Expected:

```text
client0 weight = 0.625
client1 weight = 0.375
```

This test distinguishes activation-frequency weighting from raw-count weighting.

Raw counts:

```text
10 vs 30
```

would incorrectly produce:

```text
0.25 vs 0.75
```

and must not be accepted for this method.

## 10.6 Zero-activity test

If an expert receives no valid activation contribution in a round, verify that its previous server parameters remain unchanged.

## 10.7 Shared aggregation test

Verify both expert methods use the exact same uniform shared aggregation implementation.

## 10.8 FAIR_AB config test

Add or use a normalized semantic config comparison capable of detecting unintended scientific differences.

---

# 11. Experiment Execution and Resource Safety

## 11.1 Resource inspection

Before every experiment batch inspect:

```bash
nvidia-smi
free -h
```

When useful also inspect:

```bash
ps -eo pid,ppid,user,stat,rss,cmd --sort=-rss | head -50
```

Do not interfere with external users.

Never kill, pause, renice, move, or reuse an unknown process.

## 11.2 GPU process rule

Maximum planned experiment processes per physical GPU:

```text
MAX_PLANNED_EXPERIMENTS_PER_GPU = 2
```

External users' processes do not count as one of the user's planned slots, but they must be considered when judging whether the GPU has sufficient resources.

Prefer matched resource conditions for each Direct/Activation pair.

When feasible, a paired configuration may run:

```text
Direct
+
Activation
```

on the same physical GPU concurrently, provided the two-process limit and memory constraints are respected.

Inside a process launched with:

```text
CUDA_VISIBLE_DEVICES=<physical_gpu>
```

use:

```text
device = cuda:0
```

Do not compare wall-clock time scientifically when jobs experienced different resource contention.

## 11.3 Long training must be disconnect-safe

Formal 200-round runs must not depend on the user's VS Code or SSH connection remaining open.

Preferred launcher:

```text
tmux
```

If tmux is unavailable, `nohup` may be used.

Do not install tmux autonomously.

Every serious long run must retain at least:

```text
run_id
trial_id
method
gpu_id

exact command
launch timestamp

tmux session or controller PID
Python PID when available

launcher log
output directory
exit-code file
```

## 11.4 Detached tmux pattern

A safe pattern is:

```bash
cd /home/cjq/Project/fl_moe
mkdir -p experiments/tuning_logs/detached

SESSION="<unique_session_name>"
LOG="experiments/tuning_logs/detached/${SESSION}.launcher.log"
PIDFILE="experiments/tuning_logs/detached/${SESSION}.pid"
EXITFILE="experiments/tuning_logs/detached/${SESSION}.exit_code"

tmux new-session -d -s "$SESSION" \
"bash -lc '
cd /home/cjq/Project/fl_moe

echo \$\$ > \"$PIDFILE\"

CUDA_VISIBLE_DEVICES=<gpu_id> \
/home/cjq/anaconda3/envs/fl_moe/bin/python \
<experiment_command> \
>> \"$LOG\" 2>&1

code=\$?
echo \$code > \"$EXITFILE\"
exit \$code
'"
```

Adapt quoting safely when implementing real commands.

## 11.5 Launch verification

A successful `tmux new-session` command is not enough.

After launch verify:

```text
tmux session exists
expected Python process exists
pinned interpreter is used
assigned GPU contains the process
launcher log has no immediate traceback
no duplicate run was launched
```

Useful commands:

```bash
tmux list-sessions
tmux has-session -t "<session>"
nvidia-smi
tail -n 50 "<launcher_log>"
```

## 11.6 Completion verification

A vanished tmux session does not prove successful completion.

Before declaring success verify:

```text
expected communication rounds exist
exit code == 0 when recorded
no fatal traceback
summary/metrics artifacts exist
final metrics exist
checkpoint state is consistent
```

Otherwise classify as:

```text
FAILED
INTERRUPTED
UNKNOWN
```

rather than fabricating completion.

---

# 12. Checkpoint and Resume

Checkpoint/resume may be implemented if missing.

```text
CHECKPOINT_RESUME_ALLOWED = true
```

It is infrastructure, not a scientific tuning variable.

## 12.1 Minimum checkpoint contents

Preserve enough state to resume from the exact completed communication round:

```text
global model state
completed round index
full experiment config
metric histories
method state if any
```

Also preserve optimizer/scheduler/RNG state when required by the actual framework for deterministic continuation.

## 12.2 Resume invariants

Resume must preserve:

```text
same partition
same partition hash
same seed
same model
same method
same training hyperparameters
same absolute communication-round index
```

Do not silently restart the LR schedule from zero.

Do not regenerate the partition.

Do not silently restart training from round zero if the user's intent is continuation from a valid checkpoint.

## 12.3 Direct and Activation parity

Checkpoint/resume behavior must be identical for both methods.

Checkpoint infrastructure does not count as a scientific A/B difference.

## 12.4 Resume regression test

Before relying on resume for formal experiments, perform a short deterministic equivalence test.

For example:

```text
continuous training to round N
```

versus:

```text
train to round M
save
resume
continue to round N
```

should produce equivalent training semantics under the deterministic protocol.

If current data-loader or RNG behavior prevents deterministic continuation, preserve the required RNG/sampler states instead of accepting silent divergence.

---

# 13. Workflow, Convergence, Target Classification, and Records

## 13.1 Phase 0 — Read-only audit

When this skill is first invoked on a fresh repository state:

1. run runtime preflight;
2. inspect Git state;
3. inspect remote;
4. inspect `base.py`;
5. inspect partition implementation;
6. inspect Direct method;
7. inspect Activation method;
8. inspect shared aggregation;
9. inspect local objective;
10. inspect Top-2 support;
11. inspect tests;
12. inspect checkpoint/resume capability;
13. inspect current output/logging infrastructure.

Phase 0 is read-only.

Do not modify code.

Return:

```text
what matches the skill
what differs
what is ambiguous
what requires repair
what tests are missing
recommended next actions
```

Then STOP.

Wait for explicit user approval before modifying repository code.

### Ambiguity rule

If a scientifically meaningful implementation detail is unclear or conflicts with this skill:

**ask the user before changing it.**

Do not guess and do not silently reinterpret the algorithm.

## 13.2 Phase 1 — Correctness repair

After explicit approval:

* repair confirmed correctness issues;
* add focused tests;
* add partition validation if missing;
* add logging needed for scientific verification;
* add checkpoint/resume if justified and missing.

Do not perform architecture tuning.

Do not change canonical aggregation semantics.

## 13.3 Phase 2 — Smoke tests

For each new common candidate run:

```text
2–5 rounds
```

Smoke tests exist only to verify:

```text
forward works
backward works
Top-2 works
routing counts are consistent
expert aggregation works
shared aggregation works
logging works
checkpoint works
no NaN/Inf
```

Do not interpret smoke accuracy as scientific evidence.

## 13.4 Phase 3 — Initial formal experiments

Initial candidates:

```text
B0:
    lr = 0.001
    balance = 0.0

B1:
    lr = 0.001
    balance = 0.01
```

Each candidate must contain:

```text
Direct
Activation
```

Formal duration:

```text
200 communication rounds
```

Therefore the initial complete formal set contains at most:

```text
B0 Direct
B0 Activation
B1 Direct
B1 Activation
```

provided the user has approved launching them.

Do not reject a healthy run merely because R20/R40 accuracy looks unfavorable.

Final conclusions require the declared formal budget.

## 13.5 Phase 4 — Evidence-driven shared tuning

If one initial candidate reaches Level A:

```text
freeze it
```

Do not continue tuning merely to enlarge the gap.

If neither reaches the target:

1. analyze both complete pairs;
2. identify the strongest bottleneck;
3. determine whether shared LR tuning is justified;
4. propose a small shared candidate set;
5. obtain approval where needed;
6. rerun paired Direct/Activation experiments.

Do not launch a giant blind grid.

## 13.6 Formal training budget

The current formal budget is:

```text
200 rounds
```

Codex must not autonomously change it to 300.

If both methods remain clearly unconverged at 200 rounds, report the evidence and ask the user whether to increase both methods to a larger common budget.

---

# 14. Convergence Gate

Do not classify a lucky final round as convergence.

For each full run compute:

```text
final_accuracy
best_accuracy

last5_mean

last10_mean
last10_std
last10_slope

final_test_loss
```

Also inspect routing health.

Default convergence gate:

```text
no NaN / Inf

last10_std <= 2.0 percentage points

abs(last10_slope)
<= 0.20 percentage points per round

abs(final_accuracy - last10_mean)
<= 3.0 percentage points

final_accuracy
>= best_accuracy - 5.0 percentage points

no persistent catastrophic expert routing collapse
```

Do not silently weaken these thresholds.

If the thresholds appear inappropriate, report the problem and propose a revised rule before applying it to later experiments.

A single unusual routing round is not automatically persistent collapse.

Use multiple rounds and route/participant evidence.

---

# 15. Target Levels

Primary metric:

```text
final communication-round CIFAR-10 test accuracy
```

Define:

```text
direct_acc
activation_acc

gap_pp = activation_acc - direct_acc
gap_target_pp = 3.00

gap_target_met =
gap_pp >= 3.00
```

## 15.1 Level A — exact target

Requires:

```text
direct_acc ∈ [60,65]

activation_acc ∈ [64,68]

gap_pp >= 3.00

both converged

FAIR_AB = true
```

Then:

```text
TARGET_LEVEL = A
GAP_TARGET_MET = true
STATUS = exact_target_reached
```

If both accuracy intervals are satisfied but:

```text
gap_pp < 3
```

do not call it Level A.

## 15.2 Level B — near target

If exact Level A is not reached, the near-target reference interval is:

```text
Direct      ∈ [58,67]
Activation  ∈ [62,70]
```

Always report the actual gap.

Level B is not exact success.

Use:

```text
TARGET_LEVEL = B
STATUS = near_target_not_exact
```

## 15.3 Level C

If the pair is outside the Level B tolerance:

```text
TARGET_LEVEL = C
STATUS = target_not_reached
```

## 15.4 Target distance

For useful candidate ranking define:

```text
distance(x,[L,U]) =
    0       if L <= x <= U
    L - x   if x < L
    x - U   if x > U
```

Then:

```text
gap_distance =
max(0, 3.00 - gap_pp)
```

and:

```text
total_target_distance =
    distance(direct_acc,[60,65])
  + distance(activation_acc,[64,68])
  + gap_distance
```

Among scientifically valid pairs, prefer:

1. `FAIR_AB=true`;
2. both converged;
3. Level A;
4. smaller total target distance;
5. routing health and stability;
6. simpler common configuration.

Never prefer a candidate merely because Direct was artificially degraded.

---

# 16. Logging and Experiment Records

Create or reuse:

```text
experiments/tuning_logs/
```

The new formal tuning history should be clearly separated from legacy unequal-size results.

Recommended journal:

```text
experiments/tuning_logs/EQUAL_SIZE_TUNING_JOURNAL.md
```

The journal must begin with a statement equivalent to:

```text
All experiments recorded below use the new equal-size balanced
label-Dirichlet protocol.

Legacy unequal-size experiments are historical/diagnostic only and are
not part of this formal tuning history.
```

## 16.1 Required run logging

At minimum record:

```text
algorithm / method

dataset
partition path
partition SHA256
client sample counts

num_experts
top_k

learning rate
balance_loss_weight

round
client loss
client accuracy
test loss
test accuracy

expert route counts
expert participant counts
expert client aggregation weights
```

For Activation also retain, where practical:

```text
per-client expert activation frequency
```

For Direct remember that expert weight sums may be below 1 by design.

Do not flag this as an error.

## 16.2 Compact results table

A compact `results.csv` may include:

```text
run_id
trial_id
git_commit

method

seed
deterministic

partition_path
partition_sha256
client_size

num_clients
alpha
num_experts
top_k

lr
balance_loss_weight
rounds
local_epochs
batch_size

final_acc
best_acc

last10_mean
last10_std
last10_slope

converged
fair_ab

gap_pp
gap_target_met
target_level

output_dir
status
notes
```

Unavailable fields should be blank or `NA`.

Do not fabricate them.

---

# 17. Scientific Integrity

The exact target:

```text
Direct 60–65%
Activation 64–68%
gap >= 3pp
```

is an optimization objective.

It is not permission to manufacture the desired result.

Forbidden:

```text
method-specific LR
method-specific balance coefficient
method-specific optimizer
method-specific scheduler
method-specific model
method-specific training rounds
different seeds
different partitions
different evaluation subsets
different checkpoint selection
intentional weakening of Direct
changing canonical aggregation formulas
```

The requested target-tuning workflow may use test performance to choose configurations.

This constitutes test-set-driven model selection.

Do not hide this limitation.

For stronger final scientific claims, recommend a later protocol with:

```text
training subset
+
validation subset for tuning
+
locked test set for final evaluation
```

but do not silently change the user's current requested protocol.

---

# 18. Final Freeze and Deliverables

Once a valid Level A pair is found, freeze:

```text
source code
partition
partition hash
model architecture
num_experts
top_k
shared aggregation
local objective
balance coefficient
optimizer
learning rate
batch size
local epochs
round budget
seed
deterministic setting
evaluation procedure
checkpoint procedure
```

Perform a final FAIR_AB semantic config diff.

Recommended final artifacts:

```text
experiments/tuning_logs/final/
├── FINAL_REPORT.md
├── FINAL_MODEL_SPEC.md
├── FINAL_TRAINING_WORKFLOW.md
├── final_common_config.json
├── final_direct_config.json
├── final_activation_config.json
├── final_config_diff.txt
├── direct_summary.json
├── activation_summary.json
├── comparison.json
└── final_commands.sh
```

## 18.1 FINAL_REPORT.md

Include:

```text
formal data protocol
equal-size partition verification

final model

Direct formula
Activation-frequency formula
shared aggregation formula

common training configuration

Direct final accuracy
Activation final accuracy
gap_pp

best accuracy
last10 statistics

convergence verdicts
routing health

FAIR_AB verdict
target level
gap target verdict

major tuning history
failed candidates and lessons

limitations
exact reproduction commands
```

## 18.2 FINAL_MODEL_SPEC.md

Describe the actual executable `base.py` model used for the final result.

Do not describe a model that is not implemented.

## 18.3 FINAL_TRAINING_WORKFLOW.md

Include:

```text
working directory

pinned Python executable
runtime preflight

partition verification

formal config

Direct command
Activation command

GPU assignment
tmux launch

checkpoint/resume

evaluation

output locations

convergence verification

FAIR_AB config diff
```

## 18.4 final_commands.sh

The script must reproduce the final A/B experiment from the existing environment.

It must not install packages or mutate the environment.

Every Python invocation must explicitly use:

```text
/home/cjq/anaconda3/envs/fl_moe/bin/python
```

---

# 19. Legacy Experiment Boundary

All previous experiments using unequal client sample counts are classified as:

```text
LEGACY_UNEQUAL_SIZE
```

They may be retained for:

```text
historical context
debugging history
mechanism diagnosis
explaining why equal-size partition was adopted
```

but they must not be used as the formal tuning history for the new protocol.

Do not automatically transfer conclusions such as:

```text
old best LR
old best balance coefficient
old scheduler preference
old decay horizon
old architecture preference
old final gap
old target level
```

into the new protocol.

The new formal experimental history begins with the equal-size partition.

---

# 20. Communication Behavior

At each major phase report succinctly:

```text
what was inspected
what was verified
what is wrong or ambiguous
what changed
what tests ran
what the real result was
what the next evidence-based action is
```

Never invent metrics.

Never claim training completed unless artifacts support completion.

Never silently change algorithm semantics.

Never silently change the data partition.

Never silently change Top-2 or the number of experts.

Never change Direct from all-valid denominator to active-client denominator.

Never change Activation-frequency weighting to raw-count weighting.

Never modify the runtime environment without explicit permission.

If something scientifically meaningful is unclear:

```text
ASK THE USER BEFORE MODIFYING IT
```

Do not make an assumption merely to keep working.

---

# 21. Initial Invocation Behavior

When first invoked on a new repository state:

1. use `/home/cjq/Project/fl_moe` as the local repository;
2. verify the remote repository;
3. run the pinned-runtime preflight;
4. perform Phase 0 read-only audit only;
5. inspect current `base.py`;
6. inspect the equal-size partition implementation;
7. inspect `expert_uniform_all_valid_denominator`;
8. inspect `expert_activation_frequency_weighted`;
9. inspect shared aggregation;
10. inspect standard CE training;
11. inspect Top-2 support;
12. inspect tests;
13. inspect checkpoint/resume;
14. report exact mismatches and uncertainties;
15. propose the smallest correctness actions;
16. STOP.

Do not edit repository code during this initial Phase 0.

Wait for explicit user approval before proceeding to correctness repair, smoke tests, or formal training.
