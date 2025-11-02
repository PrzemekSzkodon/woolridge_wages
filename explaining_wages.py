#load the data and import the necessary libraries
from wage_data import load_wage1_data
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df_wage1 = load_wage1_data()
df_wage1_voi = df_wage1[['lwage', 'educ', 'tenure']]

def run_regression():
    #df_wage1_voi = sm.add_constant(df_wage1)
    wage_model = smf.ols(formula = 'lwage ~ educ + tenure', data = df_wage1_voi).fit()
    print(wage_model.summary())
    return wage_model

#calls the function
wage_model = run_regression()

#calculates the residuals and then plots a histogram of the residuals
residual_values = wage_model.resid
plt.hist(residual_values, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)
plt.show()

#calculates the predicted values and then plots a scatter plot of the predicted values vs actual values
fitted_values = wage_model.fittedvalues
plt.scatter(fitted_values, df_wage1_voi['lwage'], alpha=0.6, edgecolors='black', s=20)
plt.xlabel('Fitted Values')
plt.ylabel('Actual Values')
plt.title('Fitted Values vs Actual Values')
plt.plot([fitted_values.min(), fitted_values.max()], 
         [fitted_values.min(), fitted_values.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Create high/low categories for educ and tenure
df_wage1_voi['educ_high'] = df_wage1_voi['educ'] > df_wage1_voi['educ'].median()
df_wage1_voi['tenure_high'] = df_wage1_voi['tenure'] > df_wage1_voi['tenure'].median()

# Separate residuals by high/low categories
residuals_high_educ = residual_values[df_wage1_voi['educ_high']]
residuals_low_educ = residual_values[~df_wage1_voi['educ_high']]
residuals_high_tenure = residual_values[df_wage1_voi['tenure_high']]
residuals_low_tenure = residual_values[~df_wage1_voi['tenure_high']]

# Create comprehensive plots showing categorized residuals
fig = plt.figure(figsize=(16, 10))

# 1. Histograms of residuals by educ level
ax1 = plt.subplot(2, 3, 1)
ax1.hist(residuals_low_educ, bins=20, alpha=0.7, label='Low Educ', color='blue', edgecolor='black')
ax1.hist(residuals_high_educ, bins=20, alpha=0.7, label='High Educ', color='red', edgecolor='black')
ax1.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Frequency')
ax1.set_title('Residuals Distribution: Low vs High Education')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Histograms of residuals by tenure level
ax2 = plt.subplot(2, 3, 2)
ax2.hist(residuals_low_tenure, bins=20, alpha=0.7, label='Low Tenure', color='green', edgecolor='black')
ax2.hist(residuals_high_tenure, bins=20, alpha=0.7, label='High Tenure', color='orange', edgecolor='black')
ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax2.set_title('Residuals Distribution: Low vs High Tenure')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Box plots comparing high/low educ residuals
ax3 = plt.subplot(2, 3, 3)
box_data_educ = [residuals_low_educ, residuals_high_educ]
bp1 = ax3.boxplot(box_data_educ, labels=['Low Educ', 'High Educ'], patch_artist=True)
bp1['boxes'][0].set_facecolor('lightblue')
bp1['boxes'][1].set_facecolor('lightcoral')
ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax3.set_ylabel('Residuals')
ax3.set_title('Residuals: Low vs High Education (Box Plot)')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Box plots comparing high/low tenure residuals
ax4 = plt.subplot(2, 3, 4)
box_data_tenure = [residuals_low_tenure, residuals_high_tenure]
bp2 = ax4.boxplot(box_data_tenure, labels=['Low Tenure', 'High Tenure'], patch_artist=True)
bp2['boxes'][0].set_facecolor('lightgreen')
bp2['boxes'][1].set_facecolor('wheat')
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals: Low vs High Tenure (Box Plot)')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Scatter: Residuals vs Educ (separated by educ level)
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df_wage1_voi.loc[~df_wage1_voi['educ_high'], 'educ'], 
            residuals_low_educ, alpha=0.6, label='Low Educ', color='blue', s=30, edgecolors='black')
ax5.scatter(df_wage1_voi.loc[df_wage1_voi['educ_high'], 'educ'], 
            residuals_high_educ, alpha=0.6, label='High Educ', color='red', s=30, edgecolors='black')
ax5.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax5.set_xlabel('Education')
ax5.set_ylabel('Residuals')
ax5.set_title('Residuals vs Education (Separated by Level)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Scatter: Residuals vs Tenure (separated by tenure level)
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(df_wage1_voi.loc[~df_wage1_voi['tenure_high'], 'tenure'], 
            residuals_low_tenure, alpha=0.6, label='Low Tenure', color='green', s=30, edgecolors='black')
ax6.scatter(df_wage1_voi.loc[df_wage1_voi['tenure_high'], 'tenure'], 
            residuals_high_tenure, alpha=0.6, label='High Tenure', color='orange', s=30, edgecolors='black')
ax6.axhline(y=0, color='k', linestyle='--', linewidth=2)
ax6.set_xlabel('Tenure')
ax6.set_ylabel('Residuals')
ax6.set_title('Residuals vs Tenure (Separated by Level)')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("RESIDUAL SUMMARY BY CATEGORIES")
print("="*60)
print("\nEducation Categories:")
print(f"Low Educ - Mean: {residuals_low_educ.mean():.4f}, Std: {residuals_low_educ.std():.4f}, Count: {len(residuals_low_educ)}")
print(f"High Educ - Mean: {residuals_high_educ.mean():.4f}, Std: {residuals_high_educ.std():.4f}, Count: {len(residuals_high_educ)}")
print("\nTenure Categories:")
print(f"Low Tenure - Mean: {residuals_low_tenure.mean():.4f}, Std: {residuals_low_tenure.std():.4f}, Count: {len(residuals_low_tenure)}")
print(f"High Tenure - Mean: {residuals_high_tenure.mean():.4f}, Std: {residuals_high_tenure.std():.4f}, Count: {len(residuals_high_tenure)}")
print("="*60)




