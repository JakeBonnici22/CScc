import statsmodels.api as sm
import pandas as pd
import numpy as np

df = pd.read_csv(r'cSCC_data_clean.csv')
print(df['TumourDiamater (mm)'].unique())


'''ignore the values in the column "TumourDiamater (mm)" witout "mohs"'''
df = df[df['TumourDiamater (mm)'] != 'Mohs']


# convert the column to a numeric data type
df['TumourDiamater (mm)'] = pd.to_numeric(df['TumourDiamater (mm)'])

# predictors and response variable
X = df[['Age', 'TumourDepth', 'Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)',
        'Lymphovascular (0= Absent, 1= Present)', 'Perineural  (0= Absent, 1= Present)', 'TumourDiamater (mm)']]
y = df['Recurrence (0= No recurrence, 1= Recurrence)']

# add constant term for intercept
X = sm.add_constant(X)

# fit the logistic regression model
model = sm.Logit(y, X).fit()

# print the summary of the model
print(model.summary())
