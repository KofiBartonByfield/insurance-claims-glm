# Motor Insurance Claim Frequency Modelling

## Business question

An insurer needs to price a motor policy before it has seen a single claim
from that driver. The standard actuarial answer is a **claim-frequency
model**: predict how many claims a policy is expected to make per year of
exposure, based only on information known at underwriting (driver age,
vehicle, region, bonus-malus, etc.). Get this wrong systematically and you
either lose money on the risks you underprice, or lose customers on the
risks you overprice.

This project asks: **how much does a flexible model (XGBoost) actually
improve on the Poisson GLM insurers have used for decades, once exposure is
handled properly?** GLMs remain the industry standard largely because
they're interpretable and regulator-friendly — this project quantifies what,
if anything, is being left on the table.

## Data

[freMTPL2](https://www.openml.org/d/41214) — French Motor Third-Party
Liability, ~680k policies, one underwriting year. Two linked tables:

- **freMTPL2freq**: one row per policy — exposure, claim count, driver/vehicle/region features
- **freMTPL2sev**: one row per claim — claim amount (used here for context, not modelled directly)

Fetched automatically from OpenML at runtime — no manual download needed.

## Method

1. Merge frequency and severity tables to policy level; cap known data-entry
   outliers (exposure > 1, extreme claim counts, top 0.1% of claim amounts).
2. Three models, same target (claim count, offset by exposure):
   - **Naive baseline** — single overall claim rate for every policy
   - **Poisson GLM** — one-hot encoded features, exposure as sample weight
   - **XGBoost (Poisson objective)** — exposure passed as a log-offset (`base_margin`), same target definition as the GLM
3. Evaluation: **mean Poisson deviance** (the standard metric for this
   problem — plain MAE/RMSE are misleading on count data) and a **lift
   chart**, which checks whether a model actually separates high-risk from
   low-risk policies, not just whether it gets the average right.

## Repo structure

```
insurance-claims-glm/
├── main.py              # run this — full pipeline end to end
├── requirements.txt
├── src/
│   ├── data_loader.py   # fetch + merge + clean
│   ├── features.py      # feature prep, encoding
│   ├── models.py        # baseline, Poisson GLM, XGBoost
│   └── evaluate.py      # deviance, MAE, lift chart
└── outputs/              # results.csv + lift_chart.png land here after a run
```

## How to run

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

First run needs internet access (OpenML download, ~680k rows, cached
locally afterwards). Takes a few minutes end to end, mostly XGBoost fitting.

Outputs: `outputs/results.csv` (deviance/MAE per model) and
`outputs/lift_chart.png` (observed vs predicted frequency by risk decile).

## Findings

Held-out 20% test split, seed 42. Claim frequency is ~0.10 per policy-year and
95% of policies have no claim at all, so every gain below is small in absolute
terms. Numbers are reproduced by `python findings.py`.

| Model | Poisson deviance | vs naive |
|---|---|---|
| Naive baseline | 0.3312 | — |
| Poisson GLM | 0.3214 | -3.0% |
| XGBoost (Poisson) | 0.3067 | -7.4% |

1. **XGBoost captures about 2.5x the improvement the GLM does** over a single
   flat claim rate. Both are well calibrated overall (predicted/actual claims
   0.993 and 0.995), so the difference is in *who* is ranked as risky, not the
   average.
2. **The GLM fails at the top of the risk distribution.** In the highest-risk
   decile, actual frequency is 0.365 claims per year; XGBoost predicts 0.338,
   the GLM only 0.197 (about 46% too low). At the low end the GLM overprices
   (0.058 predicted vs 0.043 actual). Actual frequency in the riskiest decile
   is 8.5x the safest. A pricing team using the GLM would systematically
   underprice the policies most likely to claim.
3. **Spline terms on the numeric features only close about a quarter of the
   gap** (deviance 0.3177 vs 0.3214 for the GLM and 0.3067 for XGBoost), so most
   of XGBoost's edge is interactions and sharp non-linearities rather than
   curved single-variable effects. A natural next step is a GLM with targeted
   interactions, which would keep the interpretability.
4. **Bonus-malus and vehicle age drive the model.** They account for 32% and
   31% of XGBoost's total gain, then vehicle brand (11%) and driver age (8%).
   Policies with BonusMalus above 100 claim at 0.376 per year against 0.080 at
   the 50 floor, nearly 5x, and 57% of policies sit exactly at 50, so the
   feature is heavily clustered.
5. **Some features add almost nothing.** `Area` carries 0.3% of the gain and
   `Density` 1.7%, consistent with area being a banded version of density.
   Drivers under 25 claim at 0.19 per year versus 0.09–0.11 for every older
   band, roughly double.

Caveats: a single train/test split with no tuning of XGBoost, so the gaps
would need cross-validation before being quoted as precise.

**A bug worth knowing about.** The first run had XGBoost scoring *worse* than
the naive baseline (deviance 0.52). Cause: when `base_margin` is used for
exposure, XGBoost ignores `base_score` in training but adds it back at
`predict()`. The auto-estimated value (~0.05) shrank every rate prediction
about 20x. Fixed in `src/models.py` by setting `base_score=1.0`; `app.py`'s
rate-times-exposure logic was already correct once that is fixed.

## Serving the model

`main.py` now also saves the trained XGBoost model to
`outputs/xgb_poisson_model.joblib`, alongside its expected feature columns
and the categorical levels seen during training. That last part matters:
a naive re-encoding of categoricals at inference time (e.g. `.astype("category")`
on a single new row) assigns category codes based only on what's present in
that row, which won't match the codes the model was trained on — `app.py`
freezes the training-time levels and reapplies them to avoid this.

```bash
python main.py                      # trains + saves outputs/xgb_poisson_model.joblib
uvicorn app:app --reload            # serves it at http://127.0.0.1:8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Exposure": 1.0, "Area": "D", "VehPower": 6, "VehAge": 5,
       "DrivAge": 35, "BonusMalus": 50, "VehBrand": "B12",
       "VehGas": "Regular", "Density": 1200, "Region": "R24"}'
```

Note: the model's raw `.predict()` output is a claim **rate**, not a count —
it was trained with exposure as a log-offset (`base_margin`), so the offset
has to be reapplied (rate × exposure) to get an expected claim count. Both
`main.py`'s evaluation and `app.py`'s response handle this consistently.

A `Dockerfile` is included to containerize the API — build it only after
running `main.py` locally, since it copies in the saved model artifact
rather than training inside the image.

## Deployment (CI/CD)

`.github/workflows/deploy.yml` deploys to an EC2 instance on every push to
`main`, using **OIDC + AWS Systems Manager** rather than a stored SSH key —
GitHub Actions assumes a short-lived AWS role and sends the deploy commands
via SSM, so port 22 never needs to be open to the internet and no long-lived
credential sits in GitHub secrets.

One-time manual AWS setup (not automatable from here):

1. Launch an EC2 instance (Ubuntu, free-tier eligible), with an IAM instance
   profile that grants `AmazonSSMManagedInstanceCore` — this lets SSM reach it.
2. Install Docker on the instance and add `ubuntu` to the `docker` group.
3. Security group: **do not open port 22 to the internet** — SSM doesn't need
   it. Only open port 8000 (or better, put it behind an ALB with TLS later).
4. Create an IAM role for GitHub OIDC (trust policy scoped to this repo) with
   permission to call `ssm:SendCommand` on that instance.
5. GitHub repo secrets: `AWS_DEPLOY_ROLE_ARN`, `AWS_REGION`,
   `EC2_INSTANCE_ID`, `API_KEY` (any string you choose — sent with requests
   as the `x-api-key` header).

## Possible extensions

- Model claim **severity** (freMTPL2sev) alongside frequency, to get a full
  pure-premium (frequency × severity) pricing model
- SHAP values on the XGBoost model to compare its implicit feature
  importance against the GLM's coefficients
- A GLM with interaction terms, to see if some of XGBoost's edge is really
  just "the GLM didn't have that interaction"
