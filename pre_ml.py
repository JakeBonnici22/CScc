import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder



df = pd.read_csv(r'cSCC_data_clean.csv')
df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.lower()
df["HistDiag"] = df["HistDiag"].str.lower()


df = df[df['ExcisionMargin (mm)'] != '<0.1']
df['TumourDiamater (mm)'] = df['TumourDiamater (mm)'].replace('Mohs', np.nan)
df['ExcisionMargin (mm)'] = df['ExcisionMargin (mm)'].replace('Mohs', np.nan)
df['TumourDiamater (mm)'] = pd.to_numeric(df['TumourDiamater (mm)'])
df['ExcisionMargin (mm)'] = pd.to_numeric(df['ExcisionMargin (mm)'])

mean_tumour_diameter = df['TumourDiamater (mm)'].mean()
mean_excision_margin = df['ExcisionMargin (mm)'].mean()
df['TumourDiamater (mm)'].fillna(mean_tumour_diameter, inplace=True)
df['ExcisionMargin (mm)'].fillna(mean_excision_margin, inplace=True)

columns_to_scale = ['TumourDiamater (mm)', 'ExcisionMargin (mm)', 'TumourDepth']
selected_data = df[columns_to_scale]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_data)
df[columns_to_scale] = scaled_data


encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['TumourStatus', 'AnatomicalLoc', 'HistDiag']])
df = df.drop(['TumourStatus', 'AnatomicalLoc', 'HistDiag'], axis=1)
df = pd.concat([df, pd.DataFrame(encoded_features.toarray())], axis=1)


# pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.max_rows', None)  # Display all rows