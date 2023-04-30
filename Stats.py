import scipy.stats as stats
import pandas as pd

df = pd.read_csv(r'cSCC_data_clean.csv')
print(df.columns)

# contingency table
contingency_table = pd.crosstab(df['Sex'], df['Recurrence (0= No recurrence, 1= Recurrence)'])


# chi-square test for independence
chi2, pval, dof, expected = stats.chi2_contingency(contingency_table)


if pval < 0.05:
    print("There is a significant association between Sex and Recurrence.")
else:
    print("There is no significant association between Sex and Recurrence.")




# split the data into two groups: with recurrence and without recurrence
recurrence = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] == 1]['Age']
no_recurrence = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] == 0]['Age']

# t-test for comparing the means of the two groups
t, pval = stats.ttest_ind(recurrence, no_recurrence)

if pval < 0.05:
    print("There is a significant difference in Age between patients with and without recurrence.")
else:
    print("There is no significant difference in Age between patients with and without recurrence.")
