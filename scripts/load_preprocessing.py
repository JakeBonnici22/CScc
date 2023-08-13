import libs

df = libs.pd.read_csv(r'cSCC_data.csv')

#
df["HospitalN"] = df.index + 1
# df.drop(df.columns[[-1]], axis=1, inplace=True)
df.iloc[:, 1] = df.iloc[:, 1].str.strip()
df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df["Sex"] = df["Sex"].str.strip()
df = df[df['AnatomicalLoc'] != 'Eyelid']
df = df[df.iloc[:,1] != 'P']
df = df[df.iloc[:, 1].notnull()]
df = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] != 2]
df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.lower()
df["HistDiag"] = df["HistDiag"].str.lower()
df['Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)'] = \
    df['Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)'].replace({1: 0, 2: 0, 3: 1})
df = df.rename(columns={'Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)':
                            'Grade (0=Well/moderately differentiated, 1=poorly differentiated)'})

df = df.drop('Sentinel lymph node biopsy  (0= Not performed, 1= Performed)', axis=1)
df = df[df['ExcisionMargin (mm)'] != '<0.1']
df['TumourDiamater (mm)'] = df['TumourDiamater (mm)'].replace('Mohs', libs.np.nan)
df['ExcisionMargin (mm)'] = df['ExcisionMargin (mm)'].replace('Mohs', libs.np.nan)
df['TumourDiamater (mm)'] = libs.pd.to_numeric(df['TumourDiamater (mm)'])
df['ExcisionMargin (mm)'] = libs.pd.to_numeric(df['ExcisionMargin (mm)'])
mean_tumour_diameter = df['TumourDiamater (mm)'].mean()
mean_excision_margin = df['ExcisionMargin (mm)'].mean()
df['TumourDiamater (mm)'].fillna(mean_tumour_diameter, inplace=True)
df['ExcisionMargin (mm)'].fillna(mean_excision_margin, inplace=True)
sex_mapping = {'M': 1, 'F': 0}
df['Sex'] = df['Sex'].map(sex_mapping)

df.to_csv('cSCC_data_clean.csv', index=False)

print("CSV file created")