'''seaborn create male female distplot'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r'cSCC_data_clean.csv')

print(df.columns)

# ax = sns.distplot(df["Age"], color="purple", hist=False, label="Total")
#
# plt.show()
#
#
# # create a scatter plot of LRD and TumourDiamater
# plt.scatter(df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)'], df['TumourDiamater (mm)'])
# plt.xlabel('LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)')
# plt.ylabel('Tumour Diameter (mm)')
# plt.show()


# plt.hist(df['Recurrence (0= No recurrence, 1= Recurrence)'  ])
# plt.xlabel('Recurrence')
# plt.ylabel('Frequency')
# plt.show()


# plt.boxplot([df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==1]['Age'],
#             df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==2]['Age'],
#             df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==3]['Age']])
# plt.xticks([1, 2, 3], ['Local', 'Regional', 'Distant'])
# plt.xlabel('LRD')
# plt.ylabel('Age')
# plt.show()
#
# plt.scatter(df['Recurrence (0= No recurrence, 1= Recurrence)'], df['TumourDepth'])
# plt.xlabel('Recurrence')
# plt.ylabel('Tumour Depth')
# plt.show()


# sns.violinplot(x='Recurrence (0= No recurrence, 1= Recurrence)', y='TumourDepth', data=df)
# plt.ylabel('Tumour Depth')
# plt.show()

# plt.bar(['No Recurrence', 'Recurrence'], [df[df['Recurrence (0= No recurrence, 1= Recurrence)']==0]['TumourDepth'].mean(),
#                                           df[df['Recurrence (0= No recurrence, 1= Recurrence)']==1]['TumourDepth'].mean()])
# plt.ylabel('Mean Tumour Depth')
# plt.show()



sns.countplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', hue='Recurrence (0= No recurrence, 1= Recurrence)', data=df)
plt.xlabel('LRD')
plt.ylabel('Count')
plt.show()

sns.countplot(x='Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)', hue='Recurrence (0= No recurrence, 1= Recurrence)', data=df)
plt.xlabel('Tumour Grade')
plt.ylabel('Count')
plt.show()


recurrence_by_gender = df.groupby(['Sex', 'Recurrence (0= No recurrence, 1= Recurrence)']).size().unstack()
recurrence_by_gender.plot(kind='bar', stacked=True)
plt.ylabel('Count')
plt.show()

sns.boxplot(x='Recurrence (0= No recurrence, 1= Recurrence)', y='Age', data=df)
plt.ylabel('Age')
plt.show()
