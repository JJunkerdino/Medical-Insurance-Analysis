# Medical Insurance: Who Becomes a High-Cost Patient?

A logistic regression (GLM) project in **R**. The goal is to find which
factors make a person's yearly medical charges fall in the **top 30%**
("high cost"), and to predict who will be high cost.

**Full analysis with code, output and charts:**
[medical_insurance_analysis.md](medical_insurance_analysis.md)

## Key results

- **Data cleaning matters:** 1,435 of the 2,772 rows (52%) are exact copies.
  After removing them, the data has **1,337 different people**.
- **Smoking is by far the strongest factor:** 272 of 274 smokers (99.3%) are
  high cost, compared with 12.1% of non-smokers.
- **Age has a curved effect:** for non-smokers, the risk stays low until the
  late 50s and then rises fast after 60. Adding age² and age³ to the model
  lowers the AIC from 742.6 to 669.7.
- **Other factors:** each extra child multiplies the odds by 1.44, and people
  in the southwest have 52% lower odds than people in the northeast. Sex has
  no clear effect.
- **Final model (stepwise AIC):** 10-fold cross-validation accuracy of
  **92.4%** (baseline 70%) and **AUC 0.911**. It finds 78.3% of the
  high-cost people and wrongly flags only 1.5% of the low-cost people.
- The full model with smoker × age and smoker × BMI interactions **cannot be
  fitted** (separation), because only 2 smokers are not high cost.

![Charges by age and smoking status](figures/plot-charges-by-age-1.png)

![Non-smokers: chance of high cost by age](figures/plot-age-effect-1.png)

## Model comparison

All models are logistic regressions. CV scores come from the same 10 folds.

| Model | Parameters | Converged | AIC | CV accuracy | CV AUC |
|---|---:|:---:|---:|---:|---:|
| M1 Full (smoker interactions) | 12 | No | 729.2 | 90.2% | 0.887 |
| M2 Reduced (no interactions) | 10 | Yes | 742.6 | 90.3% | 0.903 |
| M3 Quadratic (+ age²) | 11 | Yes | 695.0 | 92.4% | 0.911 |
| M4 Cubic (+ age³, bmi³) | 13 | Yes | 671.5 | 92.5% | 0.910 |
| **M5 Stepwise AIC (final)** | 12 | Yes | **669.7** | 92.4% | 0.911 |
| M6 bestglm (AIC) | 5 | Yes | 743.7 | 90.2% | 0.904 |
| M7 bestglm (10-fold CV) | 3 | Yes | 752.3 | 90.2% | 0.902 |

A simple rule "smokers are high cost, non-smokers are not" already gets 90.2%.
Only the models with curved age terms (M3 to M5) beat it, because they can
also flag older non-smokers.

## Steps

1. **Data cleaning:** check missing values, remove duplicate rows, and create
   the target `high_cost` (charges at or above the 70th percentile,
   $13,780.64).
2. **Exploratory analysis:** 7 charts, for example smoking by cost group,
   charges by smoking status, and charges by age.
3. **Modelling:** 7 logistic regression models: full, reduced, quadratic,
   cubic, stepwise AIC (`MASS::stepAIC`), and best subset selection by AIC and
   by cross-validation (`bestglm`).
4. **Model comparison:** AIC plus 10-fold cross-validation (accuracy and AUC).
5. **Final model:** odds ratios with 95% confidence intervals, age effect, ROC
   curve and confusion matrix.

## Project structure

```
.
├── data/
│   └── Medical_insurance.csv         # raw data (2,772 rows)
├── figures/                          # charts made when the report is knitted
├── medical_insurance_analysis.Rmd    # source: all code and text
├── medical_insurance_analysis.md     # knitted report (easy to read on GitHub)
└── README.md
```

## How to run

1. Install [R](https://cran.r-project.org/) and
   [RStudio](https://posit.co/download/rstudio-desktop/).
   The report was made with R 4.6.1.
2. Install the packages (`MASS` already comes with R):

   ```r
   install.packages(c("dplyr", "ggplot2", "scales", "bestglm", "pROC", "rmarkdown"))
   ```

3. Open `medical_insurance_analysis.Rmd` in RStudio and click **Knit**, or run:

   ```r
   rmarkdown::render("medical_insurance_analysis.Rmd")
   ```

   This makes `medical_insurance_analysis.md` and the charts in `figures/` again.

## Data

[Medical Insurance Price Prediction](https://www.kaggle.com/datasets/harishkumardatalab/medical-insurance-price-prediction)
on Kaggle. Columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`,
`charges`.

## Tools

R, dplyr, ggplot2, MASS (`stepAIC`), bestglm, pROC, R Markdown
