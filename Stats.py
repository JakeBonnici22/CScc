import scipy.stats as stats
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import shapiro


df = pd.read_csv(r'cSCC_data_clean.csv')

# Summary statistics
print(df.describe())
print()
# contingency table
contingency_table = pd.crosstab(df['Sex'], df['Recurrence (0= No recurrence, 1= Recurrence)'])
# chi-square test for independence
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)
print()
print(f"Chi-squared: {chi2:.3f}, p-value: {pval:.10f}")
if pval < 0.05:
    print("There is a significant association between Sex and Recurrence.")
else:
    print("There is no significant association between Sex and Recurrence.")

print()

# create a contingency table of death and recurrence
ct = pd.crosstab(df['Death (0=No death, 1=Death)'], df['Recurrence (0= No recurrence, 1= Recurrence)'])
# perform a chi-squared test
chi2, p, dof, expected = chi2_contingency(ct)
# print the results
print(f"Chi-squared: {chi2:.3f}, p-value: {p:.10f}")
if p < 0.05:
    print("There is a significant association between Recurrence and Death.")
else:
    print("There is no significant association between Death and Recurrence.")



# create a contingency table
contingency_table = pd.crosstab(df['ExcisionMargin (mm)'], df['Recurrence (0= No recurrence, 1= Recurrence)'])
# perform chi-squared test
chi2, p, dof, expected = chi2_contingency(contingency_table)
# print results
print(f"Chi-squared: {chi2:.3f}, p-value: {p:.10f}")
if p < 0.05:
    print("There is a significant association between ExcisionMargin (mm) and Recurrence.")
else:
    print("There is no significant association between ExcisionMargin (mm) and Recurrence.")

print()

# split the data into two groups: with recurrence and without recurrence
recurrence = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] == 1]['Age']
no_recurrence = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] == 0]['Age']
stat, p = shapiro(recurrence)
print(f'Recurrence: {stat:.3f}, p={p:.10f}')
stat, p = shapiro(no_recurrence)
print(f'No Recurrence : {stat:.3f}, p={p:.10f}')

print()

# t-test for comparing the means of the two groups
t, pval = stats.ttest_ind(recurrence, no_recurrence)
print(f"t-statistic: {t:.3f}, p-value: {pval:.10f}")
if pval < 0.05:
    print("There is a significant difference in Age between patients with and without recurrence.")
else:
    print("There is no significant difference in Age between patients with and without recurrence.")

