import pandas as pd

df = pd.read_csv(r'cSCC_data.csv')

'''Create Id column'''
df["HospitalN"] = df.index + 1


'''remove last column 'Unnamed: 17'''
df.drop(df.columns[[-1]], axis=1, inplace=True)


'''Stripping characters'''
df.iloc[:,1] = df.iloc[:,1].str.strip()


'''Remove row which has P in the second column'''
df = df[df.iloc[:,1] != 'P']


'''display the entire dataframe without truncation'''
# pd.set_option('display.max_columns', None)


'''remove rows with missing values in  nulls'''
df = df[df.iloc[:, 1].notnull()]


'''remove entire row where 'Recurrence (0= No recurrence, 1= Recurrence)' = 2'''
df = df[df['Recurrence (0= No recurrence, 1= Recurrence)'] != 2]


'''Save file as csv'''
df.to_csv('cSCC_data_clean.csv', index=False)

