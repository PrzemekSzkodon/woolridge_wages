#load the data and import the necessary libraries
from wage_data import load_wage1_data
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd

df_wage1 = load_wage1_data()
df_wage1_voi = df_wage1[['lwage', 'educ', 'tenure']]

def run_regression():
    #df_wage1_voi = sm.add_constant(df_wage1)
    wage_model = smf.ols(formula = 'lwage ~ educ + tenure', data = df_wage1_voi).fit()
    print(wage_model.summary())
    return wage_model

#calls the function
wage_model = run_regression()

# Calculate residuals and fitted values
residual_values = wage_model.resid
fitted_values = wage_model.fittedvalues

# ============================================================================
# SHEET 1: GAUSS-MARKOV ASSUMPTIONS DIAGNOSTICS
# ============================================================================
print("\n" + "="*60)
print("GAUSS-MARKOV ASSUMPTIONS DIAGNOSTICS")
print("="*60)

"""
GAUSS-MARKOV ASSUMPTIONS:
1. Linearity: E[Y|X] = Xβ (mean is linear in parameters)
2. Random sampling: (yi, xi) are i.i.d.
3. No perfect collinearity: X has full rank
4. Zero conditional mean: E[ε|X] = 0 (exogeneity)
5. Homoscedasticity: Var(ε|X) = σ² (constant variance) - CHECKED IN SHEET 2
"""

from scipy import stats as scipy_stats
from scipy.stats import chi2

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.suptitle('Gauss-Markov Assumptions Diagnostics', fontsize=16, fontweight='bold')

# 1. Residual Histogram (Normality Check - GM assumption)
"""
GM ASSUMPTION: Normality not required for OLS to be BLUE, but needed for 
inference (t-tests, F-tests). Check if residuals are approximately normal.
"""
ax1 = axes1[0, 0]
ax1.hist(residual_values, bins=30, edgecolor='black', alpha=0.7, density=True)
ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
# Overlay normal distribution
mu, sigma = residual_values.mean(), residual_values.std()
x_norm = np.linspace(residual_values.min(), residual_values.max(), 100)
ax1.plot(x_norm, scipy_stats.norm.pdf(x_norm, mu, sigma), 'r-', linewidth=2, label='Normal fit')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Density')
ax1.set_title('Residual Distribution\n(Normality Check)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, 'Check: Bell-shaped?\nSymmetric around 0?', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Q-Q Plot (Normality Check)
"""
GM/CLM ASSUMPTION: Normal errors needed for valid inference.
Q-Q plot compares sample quantiles to theoretical normal quantiles.
Points should fall on straight line if normal.
"""
ax2 = axes1[0, 1]
scipy_stats.probplot(residual_values, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, 'Good: Points on line\nBad: Curved pattern', 
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3. Residuals vs Fitted Values (Linearity Check)
"""
GM ASSUMPTION: Linearity - E[Y|X] = Xβ
If model is correctly specified, residuals should be randomly scattered.
Patterns suggest non-linearity or omitted variables.
"""
ax3 = axes1[0, 2]
ax3.scatter(fitted_values, residual_values, alpha=0.6, s=30, edgecolors='black')
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals vs Fitted\n(Linearity Check)', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, 'GM: E[ε|X] = 0\nGood: Random scatter\nBad: Curves/patterns', 
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Residuals vs Observation Order (Independence Check)
"""
GM ASSUMPTION: Independence - errors are uncorrelated
If residuals show patterns over time/order, suggests autocorrelation.
"""
ax4 = axes1[1, 0]
obs_order = np.arange(len(residual_values))
ax4.scatter(obs_order, residual_values, alpha=0.6, s=20, edgecolors='black')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Observation Order')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals vs Observation Order\n(Independence Check)', fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.text(0.05, 0.95, 'Check: Random scatter?\nPatterns = autocorrelation', 
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 5. Fitted vs Actual Values (Model Fit)
"""
Model Specification Check: How well does the model predict?
Points close to 45° line = good fit. Systematic deviations suggest misspecification.
"""
ax5 = axes1[1, 1]
ax5.scatter(fitted_values, df_wage1_voi['lwage'], alpha=0.6, s=30, edgecolors='black')
ax5.plot([fitted_values.min(), fitted_values.max()], 
        [fitted_values.min(), fitted_values.max()], 
        'r--', linewidth=2, label='Perfect Prediction')
ax5.set_xlabel('Fitted Values')
ax5.set_ylabel('Actual Values')
ax5.set_title('Fitted vs Actual\n(Model Fit Check)', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.text(0.05, 0.95, f'R² = {wage_model.rsquared:.3f}\nCheck: Points on line?', 
         transform=ax5.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. Residual Autocorrelation (Serial Correlation Check)
"""
GM ASSUMPTION: Independence - Cov(εi, εj) = 0 for i≠j
Lag-1 autocorrelation of residuals should be ≈ 0.
Large autocorrelation suggests model misspecification.
"""
ax6 = axes1[1, 2]
residuals_lag1 = residual_values[1:]
residuals_current = residual_values[:-1]
corr_lag1 = np.corrcoef(residuals_current, residuals_lag1)[0, 1]
ax6.scatter(residuals_lag1, residuals_current, alpha=0.6, s=30, edgecolors='black')
ax6.axhline(y=0, color='r', linestyle='--', linewidth=1)
ax6.axvline(x=0, color='r', linestyle='--', linewidth=1)
ax6.set_xlabel('Residuals (t-1)')
ax6.set_ylabel('Residuals (t)')
ax6.set_title('Residual Autocorrelation\n(Serial Correlation Check)', fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.text(0.05, 0.95, f'Correlation: {corr_lag1:.4f}\nShould be ≈ 0', 
         transform=ax6.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# ============================================================================
# SHEET 2: HETEROSCEDASTICITY DIAGNOSTICS AND SOLUTIONS
# ============================================================================
print("\n" + "="*60)
print("HETEROSCEDASTICITY DIAGNOSTICS AND SOLUTIONS")
print("="*60)

# Create categories for color-coding
df_wage1_voi['educ_high'] = df_wage1_voi['educ'] > df_wage1_voi['educ'].median()
df_wage1_voi['tenure_high'] = df_wage1_voi['tenure'] > df_wage1_voi['tenure'].median()

fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle('Heteroscedasticity Diagnostics and Solutions', fontsize=16, fontweight='bold')

"""
HETEROSCEDASTICITY: Violation of GM assumption Var(ε|X) = σ²
If Var(ε|X) varies with X, we have heteroscedasticity.
Consequences: OLS still unbiased, but standard errors invalid.
Solutions: Robust SE (HC3), WLS, or transformations.
"""

# Calculate correlations and BP test (done earlier but ensure available)
corr_educ = np.corrcoef(df_wage1_voi['educ'], residual_values)[0, 1]
corr_tenure = np.corrcoef(df_wage1_voi['tenure'], residual_values)[0, 1]

# 1. Residuals vs Education (Heteroscedasticity Check)
"""
CLM ASSUMPTION: Homoscedasticity - Var(ε|X) = σ² (constant)
If variance changes with educ, we have heteroscedasticity.
Look for: Funnel shapes, increasing/decreasing spread
"""
ax1 = axes2[0, 0]
ax1.scatter(df_wage1_voi['educ'], residual_values, alpha=0.6, s=30, edgecolors='black')
ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax1.set_xlabel('Education (years)')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Education\n(Heteroscedasticity Check)', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'Corr: {corr_educ:.4f}\nGood: Random scatter\nBad: Funnel shape', 
         transform=ax1.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 2. Residuals vs Tenure (Heteroscedasticity Check)
"""
CLM ASSUMPTION: Homoscedasticity - Var(ε|X) = σ²
If variance changes with tenure, heteroscedasticity present.
Affects: Standard errors become invalid (t-stats, p-values wrong)
"""
ax2 = axes2[0, 1]
ax2.scatter(df_wage1_voi['tenure'], residual_values, alpha=0.6, s=30, edgecolors='black')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Tenure (years)')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Tenure\n(Heteroscedasticity Check)', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, f'Corr: {corr_tenure:.4f}\nGood: Random scatter\nBad: Funnel shape', 
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 3. Squared Residuals vs Fitted (Heteroscedasticity Test)
"""
BROWN-PAGAN TYPE TEST: If variance changes, squared residuals correlate with X.
This plot directly visualizes heteroscedasticity.
Increasing trend = variance increases with fitted values
"""
ax3 = axes2[0, 2]
squared_residuals = residual_values ** 2
ax3.scatter(fitted_values, squared_residuals, alpha=0.6, s=30, edgecolors='black')
ax3.set_xlabel('Fitted Values')
ax3.set_ylabel('Squared Residuals')
ax3.set_title('Squared Residuals vs Fitted\n(Variance Check)', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, 'Check: Constant spread?\nTrend = heteroscedasticity', 
         transform=ax3.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Run Breusch-Pagan test for heteroscedasticity (needed for plots 4-6)
bp_test = het_breuschpagan(residual_values, wage_model.model.exog)
bp_pvalue = bp_test[1]

# 4. Breusch-Pagan Test Visualization
"""
BREUSCH-PAGAN TEST: Formal test for heteroscedasticity
H₀: Homoscedasticity (Var(ε|X) = σ²)
H₁: Heteroscedasticity (Var(ε|X) ≠ σ²)
Test statistic follows χ² distribution under H₀
"""
ax4 = axes2[1, 0]
test_result = 'Heteroscedasticity\nDetected' if bp_pvalue < 0.05 else 'Homoscedasticity\nHolds'
color_result = 'red' if bp_pvalue < 0.05 else 'green'
ax4.text(0.5, 0.6, f'BREUSCH-PAGAN TEST', ha='center', va='center', 
         fontsize=14, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.45, f'Test Statistic: {bp_test[0]:.4f}', ha='center', va='center',
         fontsize=12, transform=ax4.transAxes)
ax4.text(0.5, 0.35, f'p-value: {bp_pvalue:.4f}', ha='center', va='center',
         fontsize=12, fontweight='bold', transform=ax4.transAxes)
ax4.text(0.5, 0.2, test_result, ha='center', va='center',
         fontsize=14, fontweight='bold', color=color_result, transform=ax4.transAxes)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.text(0.5, 0.05, f'α = 0.05\n{"Reject H₀" if bp_pvalue < 0.05 else "Fail to reject H₀"}', 
         ha='center', va='center', fontsize=10, transform=ax4.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Prepare regression comparison data (run models quickly)
df_reg = df_wage1_voi.copy()
model_ols_compare = smf.ols(formula='lwage ~ educ + tenure', data=df_reg).fit()
model_rse_compare = smf.ols(formula='lwage ~ educ + tenure', data=df_reg).fit(cov_type='HC3')

# 5. Standard Errors Comparison (OLS vs Robust SE)
"""
SOLUTION 1: Robust Standard Errors (HC3)
- OLS coefficients unchanged (still unbiased)
- Standard errors adjusted for heteroscedasticity
- Valid inference even when Var(ε|X) ≠ σ²
"""
ax5 = axes2[1, 1]
se_comparison = pd.DataFrame({
    'Variable': ['Educ', 'Tenure', 'Educ', 'Tenure'],
    'SE Type': ['Standard', 'Standard', 'Robust (HC3)', 'Robust (HC3)'],
    'SE Value': [
        model_ols_compare.bse['educ'],
        model_ols_compare.bse['tenure'],
        model_rse_compare.bse['educ'],
        model_rse_compare.bse['tenure']
    ]
})
x_pos_se = np.arange(2)
width = 0.35
ax5.bar(x_pos_se - width/2, 
       [model_ols_compare.bse['educ'], model_ols_compare.bse['tenure']],
       width, label='Standard SE', alpha=0.7, color='blue')
ax5.bar(x_pos_se + width/2,
       [model_rse_compare.bse['educ'], model_rse_compare.bse['tenure']],
       width, label='Robust SE (HC3)', alpha=0.7, color='orange')
ax5.set_xlabel('Variable')
ax5.set_ylabel('Standard Error')
ax5.set_title('SE Comparison: Standard vs Robust', fontweight='bold')
ax5.set_xticks(x_pos_se)
ax5.set_xticklabels(['Educ', 'Tenure'])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')
se_diff_educ = abs(model_ols_compare.bse['educ'] - model_rse_compare.bse['educ'])
se_diff_tenure = abs(model_ols_compare.bse['tenure'] - model_rse_compare.bse['tenure'])
ax5.text(0.05, 0.95, 
         f'SE Difference:\nEduc: {se_diff_educ:.4f}\nTenure: {se_diff_tenure:.4f}\n'
         f'{"Large diff = hetero" if se_diff_educ > 0.001 or se_diff_tenure > 0.001 else "Small diff"}',
         transform=ax5.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 6. White Test for Heteroscedasticity
"""
WHITE TEST: More general test than Breusch-Pagan
Regresses squared residuals on X, X², X³, and cross products
H₀: Homoscedasticity
H₁: Heteroscedasticity (any form)
Test statistic: nR² ~ χ²(k) where k = number of regressors in White regression
"""
ax6 = axes2[1, 2]
# Prepare data for White test (up to cubic terms)
white_data = pd.DataFrame({
    'resid_sq': residual_values ** 2,
    'educ': df_wage1_voi['educ'],
    'tenure': df_wage1_voi['tenure'],
    'educ_sq': df_wage1_voi['educ'] ** 2,
    'tenure_sq': df_wage1_voi['tenure'] ** 2,
    'educ_tenure': df_wage1_voi['educ'] * df_wage1_voi['tenure'],
    'educ_cub': df_wage1_voi['educ'] ** 3,
    'tenure_cub': df_wage1_voi['tenure'] ** 3,
    'educ_sq_tenure': df_wage1_voi['educ'] ** 2 * df_wage1_voi['tenure'],
    'educ_tenure_sq': df_wage1_voi['educ'] * df_wage1_voi['tenure'] ** 2
})

# Run White test regression
white_model = smf.ols('resid_sq ~ educ + tenure + educ_sq + tenure_sq + educ_tenure + '
                     'educ_cub + tenure_cub + educ_sq_tenure + educ_tenure_sq', 
                     data=white_data).fit()

# White test statistic: n * R²
n = len(residual_values)
white_stat = n * white_model.rsquared
df_white = len(white_model.params) - 1  # Degrees of freedom (number of regressors)
white_pvalue = 1 - chi2.cdf(white_stat, df_white)

# Display results
test_result_white = 'Heteroscedasticity\nDetected' if white_pvalue < 0.05 else 'Homoscedasticity\nHolds'
color_result_white = 'red' if white_pvalue < 0.05 else 'green'

ax6.text(0.5, 0.92, 'WHITE TEST', ha='center', va='center',
         fontsize=14, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.78, f'n×R² = {white_stat:.4f}', ha='center', va='center',
         fontsize=11, transform=ax6.transAxes)
ax6.text(0.5, 0.68, f'p-value = {white_pvalue:.4f}', ha='center', va='center',
         fontsize=11, fontweight='bold', transform=ax6.transAxes)
ax6.text(0.5, 0.55, test_result_white, ha='center', va='center',
         fontsize=12, fontweight='bold', color=color_result_white, transform=ax6.transAxes)

# Display coefficients for each power
coeff_text = "COEFFICIENTS:\n"
coeff_text += f"educ:      {white_model.params['educ']:8.4f}\n"
coeff_text += f"tenure:    {white_model.params['tenure']:8.4f}\n"
coeff_text += f"educ²:     {white_model.params['educ_sq']:8.4f}\n"
coeff_text += f"tenure²:   {white_model.params['tenure_sq']:8.4f}\n"
coeff_text += f"educ×ten:  {white_model.params['educ_tenure']:8.4f}\n"
coeff_text += f"educ³:     {white_model.params['educ_cub']:8.4f}\n"
coeff_text += f"tenure³:   {white_model.params['tenure_cub']:8.4f}\n"
coeff_text += f"educ²×ten: {white_model.params['educ_sq_tenure']:8.4f}\n"
coeff_text += f"educ×ten²: {white_model.params['educ_tenure_sq']:8.4f}"

ax6.text(0.5, 0.30, coeff_text, ha='center', va='top',
         fontsize=8, transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
         family='monospace')

ax6.text(0.5, 0.02, f'df = {df_white}, H₀: Homoscedasticity', ha='center', va='bottom',
         fontsize=9, transform=ax6.transAxes)
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)
ax6.axis('off')
ax6.set_title('White Test (Cubic Terms)', fontweight='bold')

plt.tight_layout()
plt.show()

# Summary of heteroscedasticity diagnostics
print("\n" + "="*60)
print("HETEROSCEDASTICITY SUMMARY")
print("="*60)
print(f"Correlation (Residuals, Educ):   {corr_educ:.6f}")
print(f"Correlation (Residuals, Tenure): {corr_tenure:.6f}")
print(f"Breusch-Pagan Test p-value:      {bp_pvalue:.4f}")
print(f"Interpretation: {'⚠ Heteroscedasticity detected' if bp_pvalue < 0.05 else '✓ Homoscedasticity holds'}")
print("\n" + "="*60)
print("WHITE TEST DETAILS")
print("="*60)
print(f"White Test Statistic (n×R²):     {white_stat:.4f}")
print(f"White Test p-value:              {white_pvalue:.4f}")
print(f"Degrees of freedom:              {df_white}")
print(f"Interpretation: {'⚠ Heteroscedasticity detected' if white_pvalue < 0.05 else '✓ Homoscedasticity holds'}")
print("\nWhite Test Coefficients (regressing squared residuals):")
print(f"  educ:         {white_model.params['educ']:10.6f}")
print(f"  tenure:       {white_model.params['tenure']:10.6f}")
print(f"  educ²:         {white_model.params['educ_sq']:10.6f}")
print(f"  tenure²:       {white_model.params['tenure_sq']:10.6f}")
print(f"  educ×tenure:   {white_model.params['educ_tenure']:10.6f}")
print(f"  educ³:         {white_model.params['educ_cub']:10.6f}")
print(f"  tenure³:       {white_model.params['tenure_cub']:10.6f}")
print(f"  educ²×tenure:  {white_model.params['educ_sq_tenure']:10.6f}")
print(f"  educ×tenure²:   {white_model.params['educ_tenure_sq']:10.6f}")
print("="*60)

# ============================================================================
# COMPARING DIFFERENT REGRESSION METHODS
# ============================================================================
print("\n" + "="*60)
print("COMPARING DIFFERENT REGRESSION METHODS")
print("="*60)

# Prepare data
df_reg = df_wage1_voi.copy()

# Method 1: Standard OLS (already done, but recreate for comparison)
print("\n1. STANDARD OLS:")
model_ols = smf.ols(formula='lwage ~ educ + tenure', data=df_reg).fit()
print(model_ols.summary())

# Method 2: OLS with Robust Standard Errors (HC3)
print("\n2. OLS WITH ROBUST STANDARD ERRORS (HC3):")
"""
ECONOMETRIC THEORY: Robust standard errors (heteroscedasticity-consistent) allow 
valid inference even when Var(ε|X) ≠ σ². HC3 (Huber-White) estimator is preferred
for small samples. Provides correct standard errors when homoscedasticity fails.
"""
model_rse = smf.ols(formula='lwage ~ educ + tenure', data=df_reg).fit(cov_type='HC3')
print(model_rse.summary())

# Method 3: OLS with log-transformed variables
print("\n3. OLS WITH LOG-TRANSFORMED VARIABLES:")
"""
ECONOMETRIC THEORY: Log transformations can:
- Stabilize variance (reduce heteroscedasticity)
- Linearize relationships
- Interpret coefficients as elasticities
Note: lwage is already log-transformed, so we log-transform educ and tenure
"""
df_reg_log = df_reg.copy()
df_reg_log['log_educ'] = np.log(df_reg['educ'] + 1)  # +1 to handle zeros if any
df_reg_log['log_tenure'] = np.log(df_reg['tenure'] + 1)
model_log = smf.ols(formula='lwage ~ log_educ + log_tenure', data=df_reg_log).fit()
print(model_log.summary())

# Method 4: Generalized Least Squares (GLS)
print("\n4. GENERALIZED LEAST SQUARES (GLS):")
"""
ECONOMETRIC THEORY: GLS accounts for heteroscedasticity and/or autocorrelation.
Since we don't know the true variance structure, we use Feasible GLS (FGLS):
1. Estimate error variances from OLS residuals
2. Use inverse of estimated variance matrix as weights
More efficient than OLS when variance structure is correctly specified.
"""
import statsmodels.api as sm
# Estimate variance from squared residuals (FGLS approach)
residuals_sq = model_ols.resid ** 2
# Use fitted values from squared residuals regression to estimate variance
var_est = smf.ols('residuals_sq ~ educ + tenure', 
                  data=pd.DataFrame({'residuals_sq': residuals_sq,
                                   'educ': df_reg['educ'],
                                   'tenure': df_reg['tenure']})).fit().fittedvalues
var_est = np.maximum(var_est, 0.01)  # Ensure positive
weights_gls = 1 / var_est
# Normalize weights
weights_gls = weights_gls / weights_gls.max()

# Fit GLS using WLS with estimated weights (FGLS)
model_gls = smf.wls(formula='lwage ~ educ + tenure', data=df_reg, weights=weights_gls).fit()
print(model_gls.summary())
print("Note: This is Feasible GLS (FGLS) - variance structure estimated from data")

# Create comparison table
print("\n" + "="*80)
print("REGRESSION RESULTS COMPARISON")
print("="*80)

comparison_data = {
    'Method': ['OLS (Standard)', 'OLS (Robust SE)', 'OLS (Log Transformed)', 'GLS (FGLS)'],
    'β_educ': [
        model_ols.params['educ'],
        model_rse.params['educ'],
        model_log.params['log_educ'],
        model_gls.params['educ']
    ],
    'SE_educ': [
        model_ols.bse['educ'],
        model_rse.bse['educ'],
        model_log.bse['log_educ'],
        model_gls.bse['educ']
    ],
    'β_tenure': [
        model_ols.params['tenure'],
        model_rse.params['tenure'],
        model_log.params['log_tenure'],
        model_gls.params['tenure']
    ],
    'SE_tenure': [
        model_ols.bse['tenure'],
        model_rse.bse['tenure'],
        model_log.bse['log_tenure'],
        model_gls.bse['tenure']
    ],
    'R²': [
        model_ols.rsquared,
        model_rse.rsquared,
        model_log.rsquared,
        model_gls.rsquared
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

# Visual comparison
print("\n" + "="*80)
print("CREATING VISUAL COMPARISON OF REGRESSION METHODS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparison of Regression Methods', fontsize=16, fontweight='bold')

methods = comparison_df['Method']

# Plot 1: Coefficient comparison - Education and Tenure (Box plots)
"""
ECONOMETRIC INTERPRETATION: Box plots show distribution of coefficients across methods.
Shows: median (center line), quartiles (box), and variability (whiskers).
Helps assess consistency of coefficient estimates across methods.
"""
ax1 = axes[0, 0]
educ_coefs = comparison_df['β_educ'].values
educ_se = comparison_df['SE_educ'].values
# Create box plot data: coefficients with their uncertainty
np.random.seed(42)  # For reproducibility
coef_data_educ = []
for i, (coef, se) in enumerate(zip(educ_coefs, educ_se)):
    # Simulate distribution around coefficient (normal approximation)
    coef_dist = np.random.normal(coef, se, 1000)
    coef_data_educ.append(coef_dist)
bp1 = ax1.boxplot(coef_data_educ, labels=methods, patch_artist=True, widths=0.6)
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
ax1.set_ylabel('Coefficient Value')
ax1.set_title('Education Coefficient (β_educ)\nDistribution Comparison', fontweight='bold')
ax1.set_xticklabels(methods, rotation=15, ha='right')
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax1.grid(True, alpha=0.3, axis='y')
# Add point estimates
for i, coef in enumerate(educ_coefs):
    ax1.plot(i+1, coef, 'ro', markersize=8, label='Point Estimate' if i == 0 else '')

# Plot 2: Coefficient comparison - Tenure (Box plots)
ax2 = axes[0, 1]
tenure_coefs = comparison_df['β_tenure'].values
tenure_se = comparison_df['SE_tenure'].values
coef_data_tenure = []
for i, (coef, se) in enumerate(zip(tenure_coefs, tenure_se)):
    coef_dist = np.random.normal(coef, se, 1000)
    coef_data_tenure.append(coef_dist)
bp2 = ax2.boxplot(coef_data_tenure, labels=methods, patch_artist=True, widths=0.6)
for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
    patch.set_alpha(0.7)
ax2.set_ylabel('Coefficient Value')
ax2.set_title('Tenure Coefficient (β_tenure)\nDistribution Comparison', fontweight='bold')
ax2.set_xticklabels(methods, rotation=15, ha='right')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')
# Add point estimates
for i, coef in enumerate(tenure_coefs):
    ax2.plot(i+1, coef, 'ro', markersize=8)

# Plot 3: R-squared comparison
ax3 = axes[1, 0]
x_pos = np.arange(len(methods))
ax3.bar(x_pos, comparison_df['R²'], alpha=0.7, color=['blue', 'green', 'orange', 'red'], edgecolor='black')
ax3.set_xlabel('Method')
ax3.set_ylabel('R²')
ax3.set_title('R-squared Comparison', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(methods, rotation=15, ha='right')
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(comparison_df['R²']):
    ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Standard Error comparison (showing robustness)
"""
ECONOMETRIC INTERPRETATION: Standard errors comparison shows uncertainty in estimates.
Large differences between standard and robust SE indicate heteroscedasticity.
Lower SE = more precise estimates (if valid).
"""
ax4 = axes[1, 1]
x_pos_se = np.arange(2)  # educ and tenure
width = 0.2
colors = ['blue', 'green', 'orange', 'red']
for i, method in enumerate(methods):
    ax4.bar(x_pos_se + i*width, 
           [comparison_df.iloc[i]['SE_educ'], comparison_df.iloc[i]['SE_tenure']],
           width, label=method, alpha=0.7, color=colors[i])
ax4.set_xlabel('Variable')
ax4.set_ylabel('Standard Error')
ax4.set_title('Standard Errors Comparison', fontweight='bold')
ax4.set_xticks(x_pos_se + width * 1.5)
ax4.set_xticklabels(['Educ', 'Tenure'])
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Summary interpretation
print("\n" + "="*80)
print("KEY INSIGHTS FROM REGRESSION COMPARISON")
print("="*80)
print("\n1. STANDARD OLS vs ROBUST SE:")
print("   - Coefficient estimates: Same (OLS is unbiased)")
print(f"   - SE difference: Educ {abs(comparison_df.iloc[0]['SE_educ'] - comparison_df.iloc[1]['SE_educ']):.6f}, "
      f"Tenure {abs(comparison_df.iloc[0]['SE_tenure'] - comparison_df.iloc[1]['SE_tenure']):.6f}")
print("   - If SEs differ significantly → heteroscedasticity present")
print(f"   - Interpretation: {'Robust SE larger (heteroscedasticity)' if comparison_df.iloc[1]['SE_educ'] > comparison_df.iloc[0]['SE_educ'] else 'Robust SE similar (homoscedasticity)'}")

print("\n2. LOG-TRANSFORMED VARIABLES:")
print("   - Coefficients now represent elasticities")
print(f"   - R²: {comparison_df.iloc[2]['R²']:.4f} vs OLS: {comparison_df.iloc[0]['R²']:.4f}")
print("   - May reduce heteroscedasticity if variance was proportional to X")

print("\n3. GENERALIZED LEAST SQUARES (GLS/FGLS):")
print("   - Accounts for heteroscedasticity in estimation")
print(f"   - R²: {comparison_df.iloc[3]['R²']:.4f}")
print("   - More efficient than OLS when variance structure is correctly estimated")
print("   - Note: Using Feasible GLS (FGLS) since true variance structure unknown")

print("\n4. RECOMMENDATION:")
if bp_pvalue < 0.05:
    print("   ⚠ Heteroscedasticity detected → Use Robust SE or GLS")
    print("   - For inference: Use Robust SE (HC3) - always valid")
    print("   - For efficiency: Consider GLS/FGLS (if variance structure estimated well)")
else:
    print("   ✓ Homoscedasticity → Standard OLS is appropriate")
print("="*80)

"""
Completed
"""


