# PAT — the Proliferation-Adjusted Test

A one-afternoon screening test for any candidate prognostic gene in a tumour
expression cohort. It asks a single question: **does this gene still predict
outcome once you account for how fast the tumour is dividing, alongside the
standard clinical risk factors?**

For many candidate genes in neuroblastoma the answer is no. PAT makes that
answer explicit and reproducible, and returns a verdict rather than a p value.

---

## 1. What the test does

PAT compares two nested Cox proportional-hazards models for the same patients:

```
baseline :  age + stage (or risk group) + MYCN status + proliferation
full     :  baseline + candidate gene
```

The candidate gene enters as a continuous, within-cohort standardised log2
expression value. The models are compared by likelihood-ratio test, change in
Harrell's concordance index, change in AIC and change in time-dependent AUC at
3 and 5 years. A gene is called informative only if it improves the model in at
least two independent cohorts with the same direction of effect.

The proliferation covariate is the part that most published analyses omit.
PAT runs three versions of it: `CCNB2` (primary), `MKI67` (where measured) and
a ten-gene proliferation score (sensitivity analysis). A gene belonging to the
score is excluded from its own score.

## 2. Installation

```bash
pip install -r requirements.txt
```

Tested with the versions in `ENVIRONMENT.txt`. No compiled dependencies beyond
those pulled in by scikit-survival.

## 3. Input format

**Expression matrix** (`--expr`), genes in rows, samples in columns, log2 scale,
tab-separated (`.tsv`, `.txt`, `.gz`) or comma-separated:

```
gene	GSM1528894	GSM1528895	...
TRIP13	6.214	5.902	...
CCNB2	7.118	6.440	...
```

Row names must be HGNC symbols. If your platform gives probe IDs, collapse to
symbols first (we take the probe with the highest mean expression).

**Clinical table** (`--clin`), one row per sample, sample ID in the first column:

```
sample,age18,stage4,mycn,efs_t,efs_e,os_t,os_e
GSM1528894,1,1,0,375,1,735,1
```

| column | meaning |
| --- | --- |
| `efs_t`, `os_t` | time to event or censoring, in days |
| `efs_e`, `os_e` | 1 = event, 0 = censored |
| `age18` | 1 if age at diagnosis >= 18 months |
| `stage4` | 1 if INSS stage 4 (or INRG stage M) |
| `mycn` | 1 if MYCN amplified |
| `highrisk` | 1 if high risk, where stage and MYCN are unavailable |

Pass whichever covariates your cohort has with `--covariates`. PAT drops any
covariate that is constant or entirely missing and records what it used.

## 4. Running it

```bash
python pat/run_pat.py --expr example/example_SEQC_expression.tsv \
                  --clin example/example_SEQC_clinical.csv \
                  --gene TRIP13 --endpoint efs \
                  --covariates age18 highrisk \
                  --out example_output
```

Outputs:

- `PAT_<gene>_<endpoint>.csv` — one row per proliferation covariate, with
  univariable and adjusted hazard ratios, LRT, delta C, delta AIC,
  time-dependent AUC, correlation with proliferation, and a proportional-hazards flag.
- `PAT_<gene>_<endpoint>.png` — four panels: nested hazard ratios, collinearity
  with proliferation, unadjusted Kaplan-Meier split, and added discrimination.
- A provisional verdict printed to stdout.

One cohort can never return a final verdict. Run each cohort separately, then
combine the rows:

```python
from pat_core import verdict
import pandas as pd
rows = pd.concat([pd.read_csv(f) for f in files]).to_dict('records')
print(verdict(rows))
```

## 5. The decision rule

A single analysis **passes** only if all three hold:

| criterion | threshold |
| --- | --- |
| likelihood-ratio test of the nested models | p < 0.05 |
| change in Harrell's C | >= +0.005 |
| change in AIC | <= -2 |

An analysis with fewer than 50 events is not counted either way.

| verdict | condition |
| --- | --- |
| **ADDS INFORMATION** | passes in >= 2 independent cohorts, same direction |
| **PROLIFERATION SURROGATE** | fails the pass criteria and median absolute correlation with the proliferation covariate >= 0.70 |
| **NOT INFORMATIVE** | fails the pass criteria with modest collinearity (< 0.70) |
| **INDETERMINATE** | fewer than two powered analyses, or passes with inconsistent direction |

When screening many genes, apply Benjamini-Hochberg correction to the
likelihood-ratio p values within each cohort before applying the rule.

## 6. Worked example

`example/example_SEQC_expression.tsv` and `example/example_SEQC_clinical.csv` contain the SEQC
neuroblastoma cohort (GSE62564, 498 patients, 183 event-free survival events)
restricted to 16 genes, so the example runs in seconds.

```
TRIP13  univariable HR 2.38 (p = 6e-22)
        adjusted    HR 1.44 (95% CI 1.05-1.96), delta C +0.0035
        rho with proliferation 0.87
        -> PROVISIONAL: mixed across proliferation covariates; pattern consistent
           with a proliferation surrogate. Fails on replication (see paper).

FERMT2  adjusted HR 0.75, delta C +0.013, rho with proliferation 0.10
        -> PROVISIONAL PASS in this cohort.
```

Expected console output is in `example_output/`.

## 7. Limitations

PAT is a screening test, not a proof. It uses bulk expression, so it cannot
separate tumour-cell state from stromal and immune content; its proliferation
covariate is itself an imperfect proxy for mitotic rate; it models main effects
only and tests no interactions; and it is built on retrospective cohorts
profiled at diagnosis, so it says nothing about markers that only become
informative at relapse or under therapy. A gene that passes PAT has added
prognostic information in these cohorts. It has not been shown to be causal,
to be a dependency, or to be druggable.

## 8. Citation

If you use PAT, please cite: [manuscript citation to be added on acceptance].
