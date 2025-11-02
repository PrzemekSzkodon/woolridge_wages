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

# Residuals vs educ, colored by high/low educ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs educ (colored by educ level)
scatter1 = axes[0].scatter(df_wage1_voi['educ'], residual_values,
                          c=df_wage1_voi['educ_high'], cmap='coolwarm', 
                          alpha=0.6, edgecolors='black', s=30)
axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Education')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Education (colored by High/Low Educ)')
axes[0].grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=axes[0])
cbar1.set_label('High Education (> median)', rotation=270, labelpad=15)

# Residuals vs tenure (colored by tenure level)
scatter2 = axes[1].scatter(df_wage1_voi['tenure'], residual_values,
                          c=df_wage1_voi['tenure_high'], cmap='viridis',
                          alpha=0.6, edgecolors='black', s=30)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Tenure')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Tenure (colored by High/Low Tenure)')
axes[1].grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=axes[1])
cbar2.set_label('High Tenure (> median)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show()




