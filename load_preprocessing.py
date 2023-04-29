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


'''get unique values of the second column'''
# print(df.iloc[:,1].unique())


'''get count of unique values of the second column'''
# print(df.iloc[:,1].value_counts())


'''display the entire dataframe without truncation'''
pd.set_option('display.max_columns', None)

'''print rows with missing values in  nulls'''
# print(df[df.iloc[:,1].isnull()])

'''remove rows with missing values in  nulls'''
df = df[df.iloc[:,1].notnull()]
print(df)
