import pandas as pd
import numpy as np
import os
from pathlib import Path

out_dir = Path(__file__).resolve().parents[1] / "reports"
appendix_dir = out_dir / "appendix"
figures_dir = out_dir / "figures"
appendix_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)

# Load datasets
d1_path = Path(__file__).resolve().parents[1] / "data" / "dataset1.csv"
d2_path = Path(__file__).resolve().parents[1] / "data" / "dataset2.csv"

d1 = pd.read_csv(d1_path)
d2 = pd.read_csv(d2_path)

# Basic EDA printouts
with open(appendix_dir / "analysis_output.txt", "w") as f:
    f.write("=== dataset1 head ===\n")
    f.write(str(d1.head(5)) + "\n\n")
    f.write("=== dataset1 info ===\n")
    try:
        d1.info(buf=f)
    except Exception:
        f.write(str(d1.dtypes) + "\n")
    f.write("\n\n=== dataset1 describe ===\n")
    f.write(str(d1.describe(include='all')) + "\n\n")

# Parse datetimes where present
for col in ['start_time', 'rat_period_start', 'rat_period_end']:
    if col in d1.columns:
        d1[col] = pd.to_datetime(d1[col], errors='coerce')

if 'time' in d2.columns:
    d2['time'] = pd.to_datetime(d2['time'], errors='coerce')

# Drop exact duplicates
d1 = d1.drop_duplicates().reset_index(drop=True)
d2 = d2.drop_duplicates().reset_index(drop=True)

# Create derived features
if 'seconds_after_rat_arrival' in d1.columns:
    d1['rat_present_at_landing'] = np.where(d1['seconds_after_rat_arrival'] <= 0, 1, 0)
    # sensitivity window: 30 seconds
    d1['rat_present_30s'] = np.where(d1['seconds_after_rat_arrival'] <= 30, 1, 0)
else:
    d1['rat_present_at_landing'] = np.nan

# Ensure risk is integer if present
if 'risk' in d1.columns:
    try:
        d1['risk'] = d1['risk'].astype(int)
    except Exception:
        d1['risk'] = pd.to_numeric(d1['risk'], errors='coerce').astype('Int64')

# Save cleaned datasets
clean1 = Path(__file__).resolve().parents[1] / "data" / "clean_dataset1.csv"
clean2 = Path(__file__).resolve().parents[1] / "data" / "clean_dataset2.csv"
d1.to_csv(clean1, index=False)
d2.to_csv(clean2, index=False)

# Statistical tests
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

results = {}
with open(appendix_dir / "statistical_tests.txt", "w") as outf:
    outf.write("=== Statistical tests ===\n\n")
    # Chi-square / Fisher for 2x2 if possible
    if 'rat_present_at_landing' in d1.columns and 'risk' in d1.columns:
        ct = pd.crosstab(d1['rat_present_at_landing'], d1['risk'])
        outf.write("Contingency table (rat_present_at_landing x risk):\n")
        outf.write(str(ct) + "\n\n")
        try:
            if ct.shape == (2,2):
                oddsratio, p = stats.fisher_exact(ct.values)
                outf.write(f"Fisher exact p = {p:.5f}, oddsratio = {oddsratio:.3f}\n")
                results['fisher'] = (oddsratio, p)
            else:
                chi2, p, dof, expected = stats.chi2_contingency(ct)
                outf.write(f"Chi2 p = {p:.5f}, chi2 = {chi2:.3f}, dof = {dof}\n")
                outf.write("Expected counts:\n")
                outf.write(str(expected) + "\n")
                results['chi2'] = (chi2, p)
        except Exception as e:
            outf.write("Error running chi-square/fisher: " + str(e) + "\n")
    else:
        outf.write("Necessary columns for chi-square not found.\n")

    # Logistic regression
    outf.write("\n=== Logistic regression ===\n")
    needed = ['risk','seconds_after_rat_arrival','hours_after_sunset','season']
    present = all(col in d1.columns for col in needed)
    if present:
        mod_df = d1[needed + ['reward']].copy().dropna()
        outf.write(f"Model dataframe shape: {mod_df.shape}\n")
        try:
            formula = 'risk ~ seconds_after_rat_arrival + hours_after_sunset + C(season) + reward'
            logit = smf.logit(formula, data=mod_df).fit(disp=False)
            outf.write(str(logit.summary()) + "\n\n")
            params = logit.params
            conf = logit.conf_int()
            odds = np.exp(params)
            or_ci = np.exp(conf)
            outr = pd.DataFrame({'OR': odds, 'CI_low': or_ci[0], 'CI_high': or_ci[1], 'p': logit.pvalues})
            outf.write("Odds ratios and 95% CI:\n")
            outf.write(str(outr) + "\n")
            results['logit'] = outr
        except Exception as e:
            outf.write("Error fitting logistic regression: " + str(e) + "\n")
    else:
        outf.write("Not all columns for logistic regression present. Needed: " + ", ".join(needed) + "\n")

    # Interaction model (season * seconds_after_rat_arrival)
    if present:
        try:
            formula_int = 'risk ~ seconds_after_rat_arrival * C(season) + hours_after_sunset + reward'
            logit_int = smf.logit(formula_int, data=mod_df).fit(disp=False)
            outf.write("\nInteraction model summary:\n")
            outf.write(str(logit_int.summary()) + "\n")
            results['logit_interaction'] = str(logit_int.summary())
        except Exception as e:
            outf.write("Error fitting interaction model: " + str(e) + "\n")

with open(Path(__file__).resolve().parents[1] / "reports" / "README_RUN.txt", "w") as f:
    f.write("To reproduce main analysis:\n")
    f.write("1) Create venv and install requirements: pip install -r requirements.txt\n")
    f.write("2) From project root run: python src/analysis.py\n")
    f.write("Outputs will be in reports/appendix/ and figures in reports/figures/.\n")

print("Analysis complete. Cleaned files and reports saved to 'reports' directory.")
