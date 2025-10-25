# Kaggle Submission Readiness Guide

## Purpose

This document ensures our pure math solver can be seamlessly submitted to the ARC Prize 2025 Kaggle competition without architectural compromises. It defines the thin adaptation layer between our solver and Kaggle's submission format.

**Last updated:** 2025-10-24

---

## Kaggle Submission Requirements (Critical)

### 1. Output Format

**File:** `submission.json`

**Structure:**
```json
{
  "task_id_1": [
    {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
  ],
  "task_id_2": [
    {"attempt_1": [[1, 1]], "attempt_2": [[1, 1]]},
    {"attempt_1": [[2, 2]], "attempt_2": [[2, 2]]}
  ]
}
```

**Key constraints:**
- ALL task_ids from input challenges JSON must be present
- Each task must have **exactly 2 attempts** (attempt_1 and attempt_2)
- Tasks with multiple test inputs → multiple prediction objects in same order
- Each prediction is a 2D grid (list of lists) of integers 0-9

### 2. Scoring

**Per task test output:**
- If EITHER attempt matches ground truth exactly → score 1
- Else → score 0

**Final score:** Average across all task test outputs

**Implication:** Having 2 diverse attempts maximizes our chances.

### 3. Runtime Constraints

- **Time limit:** 12 hours (CPU or GPU)
- **No internet access** during execution
- **Submission:** Via Kaggle Notebook
- **External data:** Allowed if publicly available

### 4. Code Requirements

- All code in notebook OR imported from uploaded Kaggle dataset
- Pre-trained models allowed (if publicly available)
- Open-source requirement for prize winners

---

## Our Solver's Compatibility

### Current Architecture (From Anchors)

Our solver returns:

**PASS:**
```python
{
  "solver": "Φ-classes+GLUE",
  "predictions": [grid1, grid2, ...],  # One per test input
  "receipts": {
    "mdl_candidates": [
      {"P_name": "ColorMap", "num_classes": 3, ...},
      {"P_name": "Isometry", "num_classes": 5, ...},
      ...
    ],
    "chosen_candidate": {...}
  }
}
```

**UNSAT:**
```python
{
  "witness": {"class_id": 5, "reason": "no_action_matches_class_on_trains"}
}
```

### The Two-Attempt Challenge

**Problem:** Kaggle wants 2 attempts. We return 1 deterministic solution.

**Solution: Use MDL Ranking (already built-in!)**

Our Step 2 already collects ALL passing (P, Φ, A) candidates and ranks by MDL.

**Strategy:**
- attempt_1 = best candidate (lowest MDL cost)
- attempt_2 = second-best candidate (second-lowest MDL cost)
- If only 1 candidate: attempt_1 = attempt_2 (duplicate)

**Advantages:**
- Leverages existing MDL infrastructure (Decision 11)
- Provides diverse attempts when multiple solutions exist
- Zero architectural changes needed
- Maximizes score potential

### UNSAT Handling Strategy

**Problem:** What to submit when solver returns UNSAT?

**Solution: Best-Effort Guess**

**Tier 1 (Simple):**
- Use Identity transform (copy input as output)
- Works for tasks where output = input

**Tier 2 (Smarter - Future Enhancement):**
- Use the P+Φ candidate that got closest (from UNSAT witness)
- Apply it even though it doesn't pass FY
- Might get partial credit if dimension/structure match

**Tier 3 (Fallback):**
- Submit empty grid or all-zeros
- Guaranteed score 0, but satisfies format requirement

**Recommendation:** Start with Tier 1, enhance to Tier 2 after analyzing UNSAT patterns.

---

## Architecture: Submission Wrapper Pattern

### Clean Separation

```
┌─────────────────────────────────┐
│  Our Pure Math Solver           │
│  (Π / FY / GLUE)               │
│  - Returns best solution        │
│  - Generates receipts           │
│  - MDL candidates stored        │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│  Submission Generator           │
│  (Format Adapter)               │
│  - Extracts top 2 candidates    │
│  - Handles UNSAT gracefully     │
│  - Formats as Kaggle JSON       │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│  submission.json                │
│  (Kaggle Format)                │
└─────────────────────────────────┘
```

**Key principle:** The wrapper does NOT change our solver's behavior. It only adapts the output format.

---

## Implementation: Submission Generator

### File: `src/submission_generator.py`

**Purpose:** Translate solver output to Kaggle format.

**Pseudocode:**

```python
def generate_kaggle_submission(test_challenges_path, output_path="submission.json"):
    """
    Main entry point: wraps solver to produce Kaggle-format submission.

    For each task:
    1. Run our 3-step solver
    2. Extract top 2 MDL candidates (or best-effort if UNSAT)
    3. Format as attempt_1, attempt_2
    4. Handle multiple test inputs per task (ordering matters!)
    """
    with open(test_challenges_path) as f:
        tasks = json.load(f)

    submission = {}

    for task_id, task in tasks.items():
        # Run solver
        result, receipts = solve_task(task_id, task)

        # Extract predictions
        if result is not None:
            # PASS case
            attempt_1, attempt_2 = extract_top_2_attempts(result, receipts, task)
        else:
            # UNSAT case
            attempt_1, attempt_2 = best_effort_guess(task, receipts)

        # Format for Kaggle (handle multiple test inputs)
        task_submission = []
        for pred1, pred2 in zip(attempt_1, attempt_2):
            task_submission.append({
                "attempt_1": pred1,
                "attempt_2": pred2
            })

        submission[task_id] = task_submission

    # Write submission.json
    with open(output_path, "w") as f:
        json.dump(submission, f, indent=2)

    validate_submission_format(submission, tasks)  # Sanity check

    return submission


def extract_top_2_attempts(result, receipts, task):
    """
    Extract top 2 attempts from solver result.

    Strategy:
    1. If Step 2 passed and has multiple MDL candidates → use top 2
    2. Else → duplicate the single solution
    """
    predictions = result["predictions"]  # List[Grid] - one per test input

    # Check if we have multiple candidates from Step 2
    if "mdl_candidates" in receipts and len(receipts["mdl_candidates"]) >= 2:
        # Apply top 2 candidates to test inputs
        candidate_1 = receipts["mdl_candidates"][0]
        candidate_2 = receipts["mdl_candidates"][1]

        attempt_1 = apply_candidate_to_tests(candidate_1, task["test"])
        attempt_2 = apply_candidate_to_tests(candidate_2, task["test"])
    else:
        # Only one solution: duplicate it
        attempt_1 = predictions
        attempt_2 = predictions

    return attempt_1, attempt_2


def apply_candidate_to_tests(candidate, test_inputs):
    """
    Apply a specific (P, classes, actions) candidate to test inputs.

    Returns: List[Grid] - predictions for each test input
    """
    P = candidate["P"]
    classes = candidate["classes"]
    actions = candidate["actions"]

    predictions = []
    for test_input in test_inputs:
        # Apply P
        P_test = P.apply(test_input)

        # Compute Φ signatures on P(test)
        feats = phi_signature_tables(P_test)

        # Map signatures to class IDs (from training)
        # Apply class actions
        output = apply_classes_by_signature(P_test, feats, classes, actions)

        predictions.append(output)

    return predictions


def best_effort_guess(task, receipts):
    """
    Generate best-effort guess when solver returns UNSAT.

    Strategy (prioritized):
    1. Identity transform (copy input)
    2. Closest P+Φ candidate (if witness points to one)
    3. All-zeros grid (fallback)
    """
    test_inputs = [ex["input"] for ex in task["test"]]

    # Strategy 1: Identity (works for tasks where output = input)
    attempt_1 = [copy_grid(inp) for inp in test_inputs]
    attempt_2 = attempt_1  # Same guess for both attempts

    # Future: Strategy 2 - use closest candidate from UNSAT witness
    # if "closest_candidate" in receipts["witness"]:
    #     attempt_2 = apply_candidate_to_tests(receipts["witness"]["closest_candidate"], test_inputs)

    return attempt_1, attempt_2


def validate_submission_format(submission, tasks):
    """
    Sanity check: ensure submission meets Kaggle format requirements.

    Checks:
    - All task_ids present
    - Each task has list of prediction objects
    - Each prediction has attempt_1 and attempt_2
    - Each attempt is valid grid (list of lists of integers 0-9)
    - Number of predictions matches number of test inputs
    """
    for task_id, task in tasks.items():
        assert task_id in submission, f"Missing task_id: {task_id}"

        task_preds = submission[task_id]
        num_test_inputs = len(task["test"])

        assert len(task_preds) == num_test_inputs, \
            f"Task {task_id}: {len(task_preds)} predictions but {num_test_inputs} test inputs"

        for i, pred_obj in enumerate(task_preds):
            assert "attempt_1" in pred_obj, f"Task {task_id}, test {i}: missing attempt_1"
            assert "attempt_2" in pred_obj, f"Task {task_id}, test {i}: missing attempt_2"

            for attempt in [pred_obj["attempt_1"], pred_obj["attempt_2"]]:
                assert isinstance(attempt, list), "Attempt must be list (grid)"
                assert all(isinstance(row, list) for row in attempt), "Grid rows must be lists"
                assert all(0 <= val <= 9 for row in attempt for val in row), \
                    "Grid values must be integers 0-9"

    print(f"✓ Submission format validated: {len(submission)} tasks, all checks passed")
```

---

## Repository Structure

### Proposed Layout

```
/arc-agi-oo/
├── src/
│   ├── canonicalization.py       # Π (D8, OFA, idempotence)
│   ├── global_families.py        # 16 P families
│   ├── signature_builders.py     # Φ features (parity, mod, NPS, patches, etc.)
│   ├── action_inference.py       # A inference (set_color, mirror, etc.)
│   ├── solver.py                 # 3-step main solver (Global → P+Φ → LUT)
│   ├── mdl_selection.py          # MDL cost computation and tie-breaking
│   ├── receipts.py               # Receipt generation (PASS, UNSAT, candidates)
│   └── submission_generator.py   # ← Kaggle wrapper (NEW)
│
├── notebooks/
│   └── kaggle_submission.ipynb   # ← Main Kaggle submission notebook
│
├── data/
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   └── arc-agi_test_challenges.json  # Swapped by Kaggle during rerun
│
├── tests/
│   ├── test_pi_idempotence.py    # Π tests (primary-anchor §9)
│   ├── test_global_families.py   # P family tests
│   ├── test_phi_stability.py     # Φ stability tests
│   ├── test_glue_stitching.py    # GLUE tests
│   ├── test_mdl_determinism.py   # MDL tie-breaking tests
│   ├── test_submission_format.py # Validate submission.json format
│   └── test_local_scoring.py     # Score submission against solutions locally
│
├── docs/
│   ├── anchors/
│   │   ├── primary-anchor.md
│   │   ├── spec.md
│   │   ├── implementation_plan.md
│   │   └── fundamental_decisions.md
│   ├── context_index.md
│   ├── arc-agi-kaggle-docs.md
│   └── submission_readiness.md   # ← This file
│
└── README.md
```

---

## Local Development Workflow

### 1. Development Phase

```bash
# Develop solver components
python src/solver.py  # Run on individual tasks
python -m pytest tests/  # Unit tests

# Test on training set
python src/submission_generator.py \
  data/arc-agi_training_challenges.json \
  --output train_submission.json

# Validate format
python tests/test_submission_format.py train_submission.json

# Score locally
python tests/test_local_scoring.py \
  train_submission.json \
  data/arc-agi_training_solutions.json
```

### 2. Validation Phase

```bash
# Test on evaluation set
python src/submission_generator.py \
  data/arc-agi_evaluation_challenges.json \
  --output eval_submission.json

# Score
python tests/test_local_scoring.py \
  eval_submission.json \
  data/arc-agi_evaluation_solutions.json
```

### 3. Kaggle Submission Phase

**Option A: Upload src/ as Kaggle dataset**

1. Create Kaggle dataset: "arc-agi-oo-solver"
2. Upload entire `src/` folder
3. In notebook:
   ```python
   import sys
   sys.path.append('/kaggle/input/arc-agi-oo-solver/src')
   from submission_generator import generate_kaggle_submission

   generate_kaggle_submission(
       "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
       "submission.json"
   )
   ```

**Option B: Inline all code in notebook**

1. Copy-paste all .py files into notebook cells
2. Run submission generator in final cell

**Recommendation: Option A** - cleaner, easier to iterate

---

## Kaggle Notebook Template

### File: `notebooks/kaggle_submission.ipynb`

```python
# ============================================================
# Cell 1: Setup
# ============================================================
import sys
import json
from pathlib import Path

# Add our solver to path (if uploaded as Kaggle dataset)
sys.path.append('/kaggle/input/arc-agi-oo-solver/src')

# ============================================================
# Cell 2: Import our modules
# ============================================================
from solver import solve_task
from submission_generator import generate_kaggle_submission

# ============================================================
# Cell 3: Generate submission
# ============================================================
print("Starting ARC Prize 2025 submission generation...")
print("Using pure math solver (Π / FY / GLUE)")

submission = generate_kaggle_submission(
    test_challenges_path="/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    output_path="submission.json"
)

print(f"\n✓ Generated submission for {len(submission)} tasks")

# ============================================================
# Cell 4: Verify and submit
# ============================================================
# Quick sanity check
with open("submission.json") as f:
    sub = json.load(f)

total_predictions = sum(len(preds) for preds in sub.values())
print(f"Total test outputs predicted: {total_predictions}")

# Kaggle will automatically pick up submission.json
print("\n✓ Submission ready!")
```

---

## Testing & Validation Tools

### File: `tests/test_submission_format.py`

```python
"""Validate that submission.json meets Kaggle format requirements."""

import json
import sys

def validate_submission(submission_path, challenges_path):
    """
    Validate submission format against Kaggle requirements.

    Returns: (valid: bool, errors: List[str])
    """
    errors = []

    with open(submission_path) as f:
        submission = json.load(f)

    with open(challenges_path) as f:
        challenges = json.load(f)

    # Check 1: All task_ids present
    for task_id in challenges.keys():
        if task_id not in submission:
            errors.append(f"Missing task_id: {task_id}")

    # Check 2: Each task has correct structure
    for task_id, task in challenges.items():
        if task_id not in submission:
            continue

        task_preds = submission[task_id]
        num_test_inputs = len(task["test"])

        if len(task_preds) != num_test_inputs:
            errors.append(
                f"Task {task_id}: {len(task_preds)} predictions "
                f"but {num_test_inputs} test inputs"
            )
            continue

        for i, pred_obj in enumerate(task_preds):
            # Check attempt_1 and attempt_2 exist
            if "attempt_1" not in pred_obj:
                errors.append(f"Task {task_id}, test {i}: missing attempt_1")
            if "attempt_2" not in pred_obj:
                errors.append(f"Task {task_id}, test {i}: missing attempt_2")

            # Check grid format
            for attempt_name in ["attempt_1", "attempt_2"]:
                if attempt_name not in pred_obj:
                    continue

                attempt = pred_obj[attempt_name]

                if not isinstance(attempt, list):
                    errors.append(
                        f"Task {task_id}, test {i}, {attempt_name}: "
                        f"must be list, got {type(attempt)}"
                    )
                    continue

                for row_idx, row in enumerate(attempt):
                    if not isinstance(row, list):
                        errors.append(
                            f"Task {task_id}, test {i}, {attempt_name}, row {row_idx}: "
                            f"must be list, got {type(row)}"
                        )
                        continue

                    for col_idx, val in enumerate(row):
                        if not isinstance(val, int) or not (0 <= val <= 9):
                            errors.append(
                                f"Task {task_id}, test {i}, {attempt_name}, "
                                f"row {row_idx}, col {col_idx}: "
                                f"must be integer 0-9, got {val}"
                            )

    return len(errors) == 0, errors


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_submission_format.py <submission.json> <challenges.json>")
        sys.exit(1)

    valid, errors = validate_submission(sys.argv[1], sys.argv[2])

    if valid:
        print("✓ Submission format is valid!")
        sys.exit(0)
    else:
        print(f"✗ Found {len(errors)} validation errors:\n")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
```

### File: `tests/test_local_scoring.py`

```python
"""Score submission.json against solutions locally (before Kaggle submission)."""

import json
import sys

def score_submission(submission_path, solutions_path):
    """
    Compute score as per Kaggle evaluation metric.

    Returns: (score: float, task_scores: Dict[str, float])
    """
    with open(submission_path) as f:
        submission = json.load(f)

    with open(solutions_path) as f:
        solutions = json.load(f)

    task_scores = {}
    total_outputs = 0
    total_correct = 0

    for task_id, task_solution in solutions.items():
        if task_id not in submission:
            print(f"Warning: Task {task_id} not in submission")
            continue

        task_preds = submission[task_id]

        # Each task may have multiple test outputs
        for i, (pred_obj, solution_output) in enumerate(zip(task_preds, task_solution)):
            total_outputs += 1

            # Check if EITHER attempt matches
            attempt_1 = pred_obj["attempt_1"]
            attempt_2 = pred_obj["attempt_2"]

            if attempt_1 == solution_output or attempt_2 == solution_output:
                total_correct += 1
                task_scores[f"{task_id}_{i}"] = 1.0
            else:
                task_scores[f"{task_id}_{i}"] = 0.0

    overall_score = total_correct / total_outputs if total_outputs > 0 else 0.0

    return overall_score, task_scores


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_local_scoring.py <submission.json> <solutions.json>")
        sys.exit(1)

    score, task_scores = score_submission(sys.argv[1], sys.argv[2])

    print(f"\n{'='*60}")
    print(f"Overall Score: {score:.4f} ({score*100:.2f}%)")
    print(f"{'='*60}\n")

    # Breakdown by task
    correct = [k for k, v in task_scores.items() if v == 1.0]
    incorrect = [k for k, v in task_scores.items() if v == 0.0]

    print(f"Correct: {len(correct)} / {len(task_scores)}")
    print(f"Incorrect: {len(incorrect)} / {len(task_scores)}")

    if incorrect:
        print(f"\nTasks to analyze (failed):")
        for task_output in incorrect[:10]:  # Show first 10
            print(f"  - {task_output}")
        if len(incorrect) > 10:
            print(f"  ... and {len(incorrect) - 10} more")
```

---

## Summary: What Changes & What Doesn't

### ✅ NO CHANGES to Core Solver

Our pure math solver remains unchanged:
- Π/FY/GLUE principles
- 3-step algorithm (Global → P+Φ → LUT)
- MDL selection in Step 2
- Receipts generation
- All anchor document specifications

### ✅ MINIMAL ADDITION: Submission Wrapper

New components (orthogonal to solver):
- `src/submission_generator.py` (~150 lines)
- `tests/test_submission_format.py` (~100 lines)
- `tests/test_local_scoring.py` (~80 lines)
- `notebooks/kaggle_submission.ipynb` (~30 lines)

**Total new code:** ~360 lines (all in adaptation layer)

### ✅ Leverages Existing Infrastructure

- MDL candidates already stored in receipts (Decision 11)
- Top 2 candidates → attempt_1 and attempt_2 (no extra work)
- UNSAT handling via best-effort (simple fallback)

---

## Implementation Checklist

### Phase 1: Core Solver (as per anchors)
- [ ] Implement Π (canonicalization)
- [ ] Implement 16 global families P
- [ ] Implement Φ signature builders
- [ ] Implement action inference A
- [ ] Implement 3-step solver
- [ ] Implement MDL selection and tie-breaking
- [ ] Implement receipts generation

### Phase 2: Submission Wrapper
- [ ] Implement `submission_generator.py`
- [ ] Implement `extract_top_2_attempts()`
- [ ] Implement `best_effort_guess()`
- [ ] Implement `validate_submission_format()`

### Phase 3: Testing & Validation
- [ ] Write format validation tests
- [ ] Write local scoring tests
- [ ] Test on training set (400 tasks)
- [ ] Test on evaluation set (400 tasks)
- [ ] Analyze UNSAT tasks

### Phase 4: Kaggle Integration
- [ ] Create Kaggle dataset with `src/` folder
- [ ] Create Kaggle notebook
- [ ] Test submission on evaluation set via Kaggle
- [ ] Verify runtime < 12 hours
- [ ] Final submission on test set

---

## Decision: Ready to Proceed with Implementation

**Conclusion:** Submission readiness is TRIVIAL and does NOT block implementation.

We can:
1. **Build our solver exactly as specified in anchor documents**
2. **Add a thin wrapper at the end** (submission_generator.py)
3. **Test locally with full control**
4. **Submit to Kaggle with minimal friction**

**The submission layer is completely orthogonal to our pure math approach.**

**No architectural compromises needed. Proceed with implementation.**

---

**Next Steps:**
1. Start implementing core solver (`src/solver.py`, `src/global_families.py`, etc.)
2. Build submission wrapper in parallel (low priority, can be added anytime)
3. Test on training data frequently
4. Iterate toward 100% coverage on training set

**Ready to code.** 🚀
