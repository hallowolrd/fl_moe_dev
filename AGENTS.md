# FL-MoE Research Instructions

This repository is an experimental research project for federated Sparse-MoE.

The primary research objective is to make:

    experiments/expert_local_kfac_whiten_layer_projection.py

outperform:

    experiments/expert_uniform_all_valid_denominator.py

by as large a stable test-accuracy margin as possible, while preserving normal
convergence and a fair matched comparison.

Primary metric:

    Gap_Last10 =
    KFAC Last-10 mean test accuracy
    -
    Uniform Last-10 mean test accuracy

The aspirational target is:

    Gap_Last10 >= +10 percentage points

This target does NOT permit intentionally degrading Uniform, selecting an
unconverged baseline, changing the Uniform algorithm semantics, changing the
locked KFAC reference, or violating any constraint below.


# 1. Python / Conda Environment

All project commands and experiments MUST use the existing Conda environment:

    conda activate fl_moe

Before executing Python training code, verify:

    echo "${CONDA_DEFAULT_ENV:-}"
    which python
    python --version

The active Conda environment must be:

    fl_moe

Do NOT modify the Python environment.

Forbidden unless the user explicitly requests it:

- conda install
- conda update
- conda remove
- pip install
- pip uninstall
- pip upgrade
- changing Python versions
- changing PyTorch versions
- changing torchvision versions
- changing CUDA packages
- recreating or cloning the environment
- editing dependency files in order to install packages
- changing system CUDA or NVIDIA drivers
- changing Conda configuration
- changing shell configuration to work around the environment constraint

If a dependency is missing, STOP and report it to the user.
Do not install it automatically.
Do not switch to another environment as a workaround.


# 2. Resource Safety Before Any Experiment

Before launching ANY training experiment, inspect host RAM and GPU resources.

## Host RAM

Use MemAvailable rather than only "free" memory.

Example:

    awk '/MemAvailable/ {printf "MemAvailable: %.2f GiB\n", $2/1024/1024}' /proc/meminfo

If MemAvailable is less than 10 GiB:

    DO NOT START THE EXPERIMENT.

Do not intentionally create swap pressure or host-memory OOM.

## GPU

Before every launch inspect:

    nvidia-smi

Prefer also:

    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv

Inspect existing GPU processes.

Never kill another process automatically.

Do not assume a GPU is available merely because CUDA is visible.

## GPU-memory safety

For a configuration whose peak VRAM usage is known:

    required free VRAM >= known peak VRAM + 2 GiB safety margin

If peak VRAM usage is unknown:

1. Do NOT immediately launch a large parallel sweep.
2. Run one short single-process smoke experiment first.
3. Measure approximate peak VRAM.
4. Record the peak.
5. Determine safe concurrency from the measured usage.

Do not oversubscribe a GPU.

If available VRAM is insufficient for estimated peak usage plus the safety
margin:

    DO NOT START THE EXPERIMENT.

If CUDA OOM occurs:

1. Stop that failed run.
2. Record the failure.
3. Do not repeatedly retry the same unsafe configuration.
4. Reduce concurrency first.
5. If one process still OOMs, ask the user before changing any scientific parameter.

Never silently reduce batch size, model size, Fisher batch size, expert count,
MoE dimension, hidden dimension, or any other scientific parameter merely to
avoid OOM. Such a change is a different experiment.


# 3. Disk / Output Safety

Before a large sweep, check available disk space for the project/output filesystem.

Do not overwrite or delete previous experiment results.

Every run must use a unique output directory.

Do not automatically delete outputs, checkpoints, metrics.csv, summary.json,
KFAC diagnostics, partitions, or launcher logs.

If disk space appears insufficient, stop and report it instead of deleting old results.


# 4. Fixed Research Target

Unless the user explicitly changes the target, experiments must use:

- dataset: CIFAR-10
- Dirichlet alpha: 0.1
- num_clients: 10
- participation_rate: 1.0
- routing: Top-2
- backbone family: ResNet18
- model family: MoE
- deterministic execution when supported

The model must remain a ResNet18-family backbone plus an MoE architecture.


# 5. Seed Lock and Partition Lock

The experimental seed is LOCKED to:

    seed = 0

Do NOT change the seed for screening, hyperparameter tuning, architecture
search, KFAC tuning, ablation studies, or confirmation runs.

Do NOT launch seed=1, seed=2, seed=3, or any other seed automatically.

Multi-seed validation may begin only after the user explicitly authorizes it.

If any plan, Goal, script, existing configuration, or automated search proposes
a seed other than 0 before explicit authorization:

    STOP and ask the user.

Reuse the same existing CIFAR-10 alpha=0.1 seed=0 client partition.

Do not regenerate, overwrite, replace, or silently modify the matching partition.

KFAC and Uniform in a matched comparison must use the exact same partition.


# 6. Convergence Definition

For this project, convergence is defined operationally as:

    During 10 consecutive rounds, Best Test Accuracy improves by
    no more than 0.1 percentage points in total.

Equivalently, over a 10-round convergence window:

    new_best_accuracy - best_accuracy_at_window_start <= 0.1 percentage points

A final matched comparison must not be declared converged merely because the
curve is temporarily noisy.

Uniform must reach at least:

    60% test accuracy

before a final result can be considered a valid converged comparison.

A final pair must satisfy the convergence criterion for BOTH KFAC and Uniform.

There is no fixed minimum absolute KFAC accuracy threshold beyond convergence,
but a configuration that achieves a large gap only by producing poor or
pathological KFAC performance must be clearly flagged.

Do not claim a +10 point result if Uniform has not converged under this definition.


# 7. Fair KFAC vs Uniform Comparison

For every COMMON model/training change, KFAC and Uniform must use identical:

- dataset
- partition
- seed
- backbone
- MoE architecture
- number of experts
- Top-k
- initialization protocol
- optimizer
- optimizer parameter groups
- learning rates
- scheduler
- weight decay
- momentum/betas where applicable
- batch size
- local epochs
- augmentation
- balance-loss settings
- communication rounds
- all other shared training hyperparameters

Common changes must be evaluated as a matched KFAC + Uniform pair.

Do NOT give KFAC a stronger common model or stronger common training
configuration while leaving Uniform on a weaker one.

It IS explicitly allowed to search for common architectures and training
regimes that mechanistically favor KFAC over Uniform, provided both methods use
exactly the same common configuration.


# 8. Uniform Baseline Is Locked

Do NOT change the defining aggregation semantics of:

    experiments/expert_uniform_all_valid_denominator.py

The all-valid-client denominator behavior must remain intact.

Allowed changes to the Uniform file are limited to necessary compatibility
changes caused by shared model/interface changes.

Do NOT change denominator semantics, expert aggregation weighting semantics, or
the intended baseline algorithm.

Do not deliberately make Uniform unstable, weaker, or slower to manufacture a larger gap.


# 9. ResNet18 Backbone Boundary

The backbone must remain ResNet18-family.

Allowed:

- the existing ResNet18-GN implementation
- GroupNorm-related hyperparameters
- CIFAR small-image stem behavior
- zero-init residual configuration when already supported

Forbidden:

- ResNet34
- ResNet50
- WideResNet
- ConvNeXt
- ViT or other non-ResNet18 backbones
- changing ResNet block activation functions
- changing ResNet channel widths
- changing the number of ResNet18 blocks
- introducing additional backbone-side projection or LayerNorm modules after the backbone

The current MoE head may keep and use its existing feature adapter / feature
normalization as part of the existing MoE implementation. Do not add new
backbone-side projection/LN modules outside that existing MoE-head boundary.

If a proposed change makes it unclear whether the backbone is still ResNet18,
STOP and ask the user.


# 10. Router Boundary

PHASE 1 router architecture is LOCKED to the current router family:

    Linear -> Softmax -> Top-2

Top-2 must remain Top-2.

Do NOT introduce in Phase 1:

- cosine router
- prototype router
- clustering router
- learned routing prototypes
- alternative router architectures

Router hyperparameters may be tuned when they do not change the router family.

If normal architecture/hyperparameter search reaches a clear plateau, a new
router family may be considered only after asking the user and receiving
explicit permission.

Do NOT introduce Top-k probability renormalization.


# 11. MoE Architecture Search Boundary

Expert count MAY be changed.

MoE architecture MAY be adjusted while preserving the ResNet18 + MoE requirement.

Allowed examples include:

- num_experts
- expert hidden dimension
- MoE representation dimension where supported by the existing MoE head
- expert MLP depth
- expert residual structure
- expert activation
- shared classifier head
- expert-specific classifier head
- other MoE-head changes that do not violate the locked ResNet18 boundary
- existing MoE-head feature adapter / feature normalization dimensions when needed by the MoE architecture

A shared classifier is explicitly allowed.

Every common architecture change must be applied identically to KFAC and Uniform.


# 12. Local Training Hyperparameters

Optimizer MAY be changed.

Allowed candidates may include, when already available in the environment:

- SGD
- Adam
- AdamW

Do not install dependencies to obtain an optimizer.

Parameter-group-specific learning rates ARE allowed, including different
learning rates for shared/backbone, router, and expert parameter groups.

Learning-rate schedulers ARE allowed.

local_epochs MAY be searched.

batch_size MAY be searched.

Balance-loss weight and scheduling MAY be searched.

Do NOT retry:

    learning_rate = 0.01

It has already been observed to fail to learn in this project and should not
consume further search budget unless the user explicitly reopens it.

When using non-SGD optimizers, schedulers, or parameter-group-specific learning
rates, do not silently change the KFAC reference or projection math.

If KFAC pseudo-gradient reconstruction requires an ambiguous interpretation of
the effective expert learning rate, STOP and ask the user before changing
method semantics.


# 13. Data Augmentation Boundary

The current default research regime is:

    train_augmentation = none

Keep data augmentation OFF unless later evidence motivates reopening the
existing standard augmentation ablation.

The existing supported modes "none" and "standard" may be compared if the
user/plan explicitly reopens augmentation.

Do NOT introduce:

- CutMix
- MixUp
- RandAugment
- AutoAugment
- ColorJitter
- other new augmentation families

Do not add augmentation libraries or dependencies.


# 14. KFAC Reference Is Completely Locked

The current KFAC reference definition is COMPLETELY LOCKED.

Preserve:

- the current self-included reference
- the current reference weighting definition
- the current valid-client/reference semantics

Forbidden:

- Leave-One-Out reference
- excluding self from the reference
- changing reference weighting
- redefining the reference formula
- replacing the reference with another consensus direction

Do not change the reference even if another reference appears likely to improve the gap.


# 15. KFAC Route Replay

Strict training-route replay MAY be enabled or disabled as an ablation/search variable.

When strict replay is enabled, preserve:

- sample-occurrence-level replay
- replay Top-k expert indices
- replay Top-k probabilities
- local_epochs > 1 keeps each processed occurrence separately

Do not silently replace strict occurrence-level replay with last-route-only replay.


# 16. KFAC Phase 1: Projection Formula Locked

PHASE 1 is limited to:

- common architecture search
- common training hyperparameter search
- existing KFAC hyperparameter tuning
- route-replay on/off
- correction-cap experiments
- diagnostic improvements that do not change update semantics

The KFAC projection/reference mathematical formula must NOT be changed during Phase 1.

Allowed KFAC-only tuning includes:

- relative_damping
- max_whitening_gain
- minimum_kfac_samples
- correction cap and its hyperparameters
- other existing numerical KFAC hyperparameters that do not redefine the projection/reference formula

Correction cap experiments are explicitly allowed.

KFAC-only changes do not require rerunning Uniform when an exact matching
Uniform common configuration already exists.


# 17. KFAC Phase 2 Requires User Permission

If architecture + hyperparameter search reaches a clear plateau and further
progress appears to require changing KFAC projection mathematics:

    STOP.

Summarize:

- current best configuration
- current Gap_Last10
- evidence that Phase 1 has plateaued
- relevant KFAC diagnostics
- the exact proposed mathematical change
- why it may help
- risks to interpretation/fairness

Then ASK the user for explicit permission.

Do NOT autonomously implement margin projection, alternative conflict criteria,
new whitened-space aggregation equations, new projection formulas, new
reference formulas, or other mathematical changes to the core KFAC method
before explicit approval.


# 18. KFAC Diagnostics

KFAC experiments should preserve and analyze, where available:

- KFAC coverage
- valid/fallback client-layer counts
- conflict rate
- projection rate
- mapped correction ratio
- whitening round-trip error
- training route counts
- replay Fisher route counts
- active clients per expert
- route distribution
- effective expert utilization
- fallback reasons

If strict replay is enabled, training route counts and replay Fisher route
counts must match according to replay semantics.

If they do not match, treat it as an implementation/debugging issue and do not
continue a long run blindly.

If KFAC gains accuracy while conflict/projection rates remain near zero,
explicitly report that the observed advantage cannot automatically be
attributed to the KFAC projection step.


# 19. Experimental Search Discipline

Do not perform uncontrolled Cartesian-product sweeps.

Prefer hypothesis-driven search and successive halving.

Typical evaluation points:

    ~50 rounds
    ~100 rounds
    full run / convergence confirmation

Do not automatically run every candidate for the maximum number of rounds.

Because project convergence is defined by a plateau criterion rather than a
fixed round count, monitor:

- Best Test Accuracy
- improvement over the previous 10-round window
- current convergence status
- Uniform >= 60% requirement

Do not stop a promising configuration solely because it has not plateaued by a
particular communication round.


# 20. Existing Results Must Be Reused

Before launching a new experiment:

1. Search existing output directories.
2. Search experiment registry / leaderboard if present.
3. Inspect config.json and method-specific config.
4. Determine whether the exact configuration has already been completed.
5. Determine whether a matching Uniform result already exists.

Do not rerun an identical completed configuration unless the user explicitly requests replication.

For a KFAC-only hyperparameter change, reuse an exact matching completed Uniform result when valid.

For partially completed or failed runs, report their state before deciding whether to rerun.


# 21. Experiment Provenance

Every scientific experiment must record or preserve:

- git commit hash
- git dirty/clean status
- exact command
- complete common configuration
- KFAC-specific configuration when applicable
- algorithm name
- seed
- partition path
- output directory
- resource/GPU assignment when practical

Before a sweep, run:

    git rev-parse HEAD
    git status --short

Do not silently mix results from different source revisions.

If the working tree is dirty, record that fact.

Never automatically run destructive Git commands such as:

    git reset --hard
    git checkout -- .
    git clean

or equivalent destructive operations.


# 22. Code Modification Discipline

Make the smallest change necessary to test the current hypothesis.

Avoid unrelated refactoring.

Do not reformat large unrelated portions of files.

Do not rename existing algorithms, output fields, CLI arguments, or diagnostic
fields without a clear need.

For KFAC-specific changes, prefer changing the KFAC experiment implementation
rather than shared behavior unless the shared change is genuinely required.

For common model/training changes, shared implementation may be modified only
when the change affects KFAC and Uniform identically.

If an implementation choice is ambiguous and different interpretations would
change scientific semantics:

    ASK THE USER BEFORE IMPLEMENTING IT.


# 23. Validation After Code Changes

Before launching a long experiment after Python code changes:

1. Run syntax/compile checks.
2. Verify CLI parsing if CLI options changed.
3. Run a minimal smoke test when practical.
4. Verify outputs are created correctly.
5. Verify KFAC and Uniform use the intended matched common configuration.
6. Verify seed=0 and the correct partition.
7. Re-check host RAM and GPU memory before the real run.

A smoke test is not a scientific result and must not be placed in the final leaderboard.


# 24. Result Ranking and Validity

Primary ranking metric:

    Gap_Last10 = KFAC_Last10 - Uniform_Last10

Also report:

- KFAC Last10
- Uniform Last10
- KFAC Best
- Uniform Best
- Gap Best
- convergence status
- convergence round/window
- KFAC diagnostics
- route/expert-utilization diagnostics

A valid final candidate must satisfy:

1. KFAC is converged under the 10-round / 0.1-point plateau criterion.
2. Uniform is converged under the same criterion.
3. Uniform reaches at least 60% test accuracy.
4. The comparison is a fair matched common configuration.
5. Uniform algorithm semantics are unchanged.
6. KFAC reference semantics are unchanged.

The +10 percentage-point target is aspirational.

If the available search budget does not reach +10 points, report the true best
valid converged configuration rather than manipulating the experiment to hit the target.


# 25. Codex Autonomy Boundary

Codex MAY autonomously:

- inspect existing results
- build/update an experiment registry
- analyze diagnostics
- tune allowed common hyperparameters
- tune allowed KFAC hyperparameters
- vary expert count
- vary allowed MoE-head architecture
- test shared classifier variants
- test allowed optimizers
- test parameter-group learning rates
- test learning-rate schedulers
- search local_epochs
- search batch size
- search balance-loss settings
- toggle strict route replay
- test correction caps
- launch fair matched KFAC/Uniform experiments when resource checks pass
- use successive halving
- stop clearly unpromising runs according to an explicit documented rule

Codex MUST STOP AND ASK THE USER before:

- changing seed
- regenerating/replacing the locked partition
- introducing a new router architecture
- changing Top-2 routing
- adding Top-k probability renormalization
- changing KFAC reference semantics
- changing KFAC projection mathematics in Phase 2
- changing Uniform aggregation semantics
- violating the ResNet18 boundary
- adding new data-augmentation families
- installing/upgrading/removing dependencies
- modifying the Conda/Python/CUDA environment
- killing other users' or unknown GPU processes
- changing scientific parameters solely to work around OOM
- taking destructive Git/filesystem actions
- making any scientifically meaningful choice whose semantics are ambiguous


# 26. Explicitly Forbidden Strategy

Do NOT use warm-up followed by freezing the shared path or backbone as a gap
engineering strategy.

Do not freeze the shared/backbone path after warm-up unless the user explicitly
reopens this idea in a later instruction.


# 27. Experiment Scheduler Rule

Only one agent/process should act as the experiment scheduler.

Analysis agents may work in parallel, but they must not independently launch
overlapping GPU sweeps.

Before each launch, the scheduler must check:

- active Conda environment
- MemAvailable >= 10 GiB
- GPU memory
- GPU utilization
- GPU processes
- existing result registry
- duplicate configuration status
- output path
- exact command/config pair
- seed=0
- locked partition

Do not start duplicate jobs.

Do not maximize concurrency at the expense of OOM risk or severe resource contention.


# 28. Stop-and-Ask Conditions

STOP and ask the user when:

- required dependencies are missing
- RAM is below 10 GiB available
- GPU memory is insufficient
- a single-process configuration repeatedly OOMs
- partition metadata is inconsistent
- seed would need to change
- a new router family appears necessary
- Phase 1 appears plateaued and KFAC math changes are proposed
- Uniform semantics would need to change
- KFAC reference semantics would need to change
- a destructive action would be required
- a scientifically meaningful implementation decision is ambiguous

When uncertain, preserve existing scientific behavior rather than silently changing it.
