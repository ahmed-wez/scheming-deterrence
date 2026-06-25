# Plan Amendment 1 — Compute Reduction

**Status: ACTIVE.** This supersedes the original plan's compute assumptions
wherever they conflict. From this point on, every phase should pull its
parameters from `configs/study_design.yaml`, not the hardcoded numbers in the
original plan document (5 seeds, 300 interactions, 1000-interaction decay
test, 4-condition mechanism analysis, simultaneous Qwen replication).

Revised budget: **~170 GPU-hours, ~$510** (vs. original ~490 hrs / ~$1,470).
~65% cost reduction.

## Decision table

| What changed | Original | Revised | GPU-hrs saved | Verdict |
|---|---|---|---|---|
| Seeds per condition | 5 | 3 | ~40% of training compute | Cut |
| Qwen 2.5 7B replication | Run alongside primary | Deferred to revision | ~200 hrs | Defer |
| Measurement interactions/condition | 300 | 200 | ~33% of inference | Cut |
| Decay test scope | 1000 interactions | 500 interactions | ~12 hrs | Reduce |
| Mechanism analysis conditions | All 4 (A,B,C,D) | A & D only | ~15 hrs | Streamline |
| 2×2 behavioral design | — | **Unchanged** | — | Keep |
| Belief vs. value experiment | — | **Unchanged** | — | Keep |
| Scheming classifier validation rigor | — | **Unchanged** | — | Keep |

## Phase-by-phase impact

### Phase 1.3 & 3.3 — Seeds: 5 → 3 per condition
Affects organism creation and belief-embedding training. Going forward, train
3 seeds, not 5. **Caveat worth keeping in mind:** this reduces statistical
power for the seed random-effect in the ANOVA, particularly for detecting the
Stated×Actual interaction (the subtlest effect). If a result is *only*
significant at 5 seeds and not 3, that's a real signal it's too fragile to
lean on for the paper's headline claim — treat it as a robustness flag, not
something to chase by adding seeds back in later.

**Nothing built yet for this phase** — no code changes needed today, this just
changes the seed count when we get to Phase 1.3.

### Phase 4.1–4.2 — Measurement interactions: 300 → 200 per condition
Proposed rebalanced split (keeps the same ~50/33/17% ratio as the original
150/100/50 split):
- Category 1 (Direct Scheming Opportunities): 100
- Category 2 (Ambiguous Opportunities): 67
- Category 3 (Control Tasks): 33

This is encoded in `study_design.yaml` and easy to change there if you'd
rather weight it differently — nothing in code depends on the exact split
being 50/33/17.

**Nothing built yet for this phase** — no code changes needed today.

### Phase 6.1 — Qwen 2.5 7B Instruct replication: deferred to revision
The repo already supports this (the `replication` target in
`configs/models.yaml`). **Action taken today:** `scripts/01_download_model.py`
now refuses to download a target marked `deferred_to_revision` unless you pass
`--confirm-deferred` explicitly. This means neither Kaggle account can
accidentally burn the ~200 GPU-hours on Qwen before it's actually time —
you'd have to deliberately override it.

When a reviewer asks for the replication later, override and run it:
```bash
python scripts/01_download_model.py --target replication --confirm-deferred
```

### Phase 6.3 — Decay test: 1000 → 500 interactions, rolling window 100 → 50
Still run on Condition B (high stated / low actual) and Condition D, so the
comparison (does B drift upward while D stays flat) is still apples-to-apples.
Halving the window size keeps the curve's shape resolvable while halving
inference cost.

### Phase 5 — Mechanism analysis: streamlined to Condition A & D only
**Important nuance worth flagging explicitly, since the screenshot's "core
claims still supported" verdict needs one caveat:** the activation-steering
causal test (5.4) was *already* an A→D comparison in the original plan, so
that core claim — "the audit-belief representation causally suppresses
scheming" — is fully preserved.

What you lose: 5.1's sub-probes for disentangling "stated-only" vs.
"actual-only" internal representations need at least 3 conditions to separate
those two factors (A is low/low, D is high/high — moving from A to D changes
both at once, so you can't attribute the representational shift to stated
belief specifically vs. actual consequence specifically). With only A and D,
you can show *that* deterrence has an internal representation and that it's
causal, but not *cleanly decompose* which factor drives the representation —
mirroring exactly the stated-vs-actual decomposition question the behavioral
2×2 answers, just not at the mechanism level too.

**Cheap fallback if a reviewer pushes on this:** Phase 4 will already have
logged full generations for Conditions B and C. Extending the probes to them
later is just a forward pass on text you already have — not new generation —
so it's a much cheaper addition than redoing this from scratch.

### Unaffected — build as originally planned, full rigor
- **Phase 0.4, scheming classifier** (kappa > 0.7, accuracy > 90%) — this
  hasn't been built yet (still pending), and the compute-reduction plan
  explicitly keeps it as-is. Don't cut corners here regardless of compute
  pressure — a wrong classifier invalidates everything downstream.
- **Phase 2, the 2×2 design itself** — unchanged.
- **Phase 4.4, belief vs. value experiment** — unchanged; this is flagged as
  the paper's most novel finding.

## Concrete repo changes made on this date
- Added `configs/study_design.yaml` — single source of truth for seeds,
  interaction counts, decay scope, and mechanism conditions.
- Added `status` field to each model in `configs/models.yaml`.
- Added a deferred-target guard to `scripts/01_download_model.py`.
- Updated `README.md` with a pointer to this document.
