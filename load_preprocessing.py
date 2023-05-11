import pandas as pd
import numpy as np

df = pd.read_csv(r'cSCC_data.csv')
'''Create Id column'''
df["HospitalN"] = df.index + 1


'''remove last column 'Unnamed: 17'''
df.drop(df.columns[[-1]], axis=1, inplace=True)


'''Stripping characters'''
df.iloc[:,1] = df.iloc[:,1].str.strip()
df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df["Sex"] = df["Sex"].str.strip()

'''Remove row which has P in the second column'''
df = df[df.iloc[:,1] != 'P']


'''display the entire dataframe without truncation'''
# pd.set_option('display.max_columns', None)


'''remove rows with missing values in  nulls'''
df = df[df.iloc[:, 1].notnull()]


'''remove entire row where 'Recurrence (0= No recurrence, 1= Recurrence)' = 2'''
df = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] != 2]


df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.lower()
df["HistDiag"] = df["HistDiag"].str.lower()

print(df.columns)


df = df[df['ExcisionMargin (mm)'] != '<0.1']
df['TumourDiamater (mm)'] = df['TumourDiamater (mm)'].replace('Mohs', np.nan)
df['ExcisionMargin (mm)'] = df['ExcisionMargin (mm)'].replace('Mohs', np.nan)
df['TumourDiamater (mm)'] = pd.to_numeric(df['TumourDiamater (mm)'])
df['ExcisionMargin (mm)'] = pd.to_numeric(df['ExcisionMargin (mm)'])

mean_tumour_diameter = df['TumourDiamater (mm)'].mean()
mean_excision_margin = df['ExcisionMargin (mm)'].mean()
df['TumourDiamater (mm)'].fillna(mean_tumour_diameter, inplace=True)
df['ExcisionMargin (mm)'].fillna(mean_excision_margin, inplace=True)


'''Save file as csv'''
df.to_csv('cSCC_data_clean.csv', index=False)

