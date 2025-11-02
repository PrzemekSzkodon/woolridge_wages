import matplotlib.pyplot as plt
from wage_data import load_wage1_data

df_wage1 = load_wage1_data()

print(df_wage1.describe())
print(df_wage1.info())
print(df_wage1.columns.tolist())

for i in ['lwage', 'educ', 'tenure']:
    plt.hist(df_wage1[i], bins=30)
    plt.title(i)
    plt.xlabel(i)
    plt.ylabel('Value')
    plt.show()





