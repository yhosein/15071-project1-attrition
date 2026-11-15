# Employee Attrition Prediction

15.071 Project 1 (Predictive AI) — predicting employee attrition to support workforce planning decisions.

## Problem

**Stakeholder:** HR / People Operations leadership.
**Decision:** Which employees are at elevated risk of leaving, and where to prioritize retention efforts (compensation, role changes, workload) before attrition happens.
**Opportunity:** Replacing an employee costs a multiple of their salary in recruiting, onboarding, and lost productivity. A model that flags at-risk employees early lets HR intervene selectively instead of applying blanket retention measures.

## Data

`data/HR-Employee-Attrition.csv` — IBM HR Analytics Employee Attrition & Performance dataset (1,470 employees, 35 features, 16.1% attrition rate). Source: Kaggle (pavansubhasht/ibm-hr-analytics-attrition-dataset); this copy mirrored from [pplonski/datasets-for-start](https://github.com/pplonski/datasets-for-start).

## Results

**Benchmark:** a non-technical business rule ("flag everyone working overtime") already catches 30/47 leavers in the test set at a budget of 76 interventions — but it has a hard ceiling: 36% of employees who leave aren't working overtime, so the rule can *never* catch them, no matter how it's applied.

**Model:** a tuned ensemble (logistic regression + random forest + gradient boosting) catches 33/47 at the same budget, including 6 of the 17 "rule-invisible" leavers the overtime rule can never reach.

**Is that margin real?** A 3,000-resample bootstrap on the test set shows the ensemble beats the rule in 68% of resamples (95% CI for the true-positive edge: [-4, +8] — includes zero, so this is not statistically airtight on its own).

**Why it's still the right call for HR:**
1. **Same cost either way** — the model and the rule spend the identical budget on the identical number of interventions; only *who* gets picked differs.
2. **Positive expected value, bounded downside** — the mean net-value edge is positive (~$30K per cycle at this budget), and even the unlucky end of the range isn't close to wiping out the program's ~$390K+ estimated value.
3. **This is a repeated, recurring decision** (e.g., an annual retention cycle) — a positive-expected-value choice made every cycle should pay off on average over time, even though any single cycle is uncertain.
4. **The structural argument stands regardless of the margin:** a single-variable rule is mathematically capped at ~64% recall here; ML is the only approach that can reach beyond it.

Full analysis: [notebooks/02_modeling.ipynb](notebooks/02_modeling.ipynb).

## Structure

```
data/           raw dataset
notebooks/      EDA, modeling, evaluation
```

## Setup

```bash
pip install -r requirements.txt
```
