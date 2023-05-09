import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_profiling

df = pd.read_csv(r'cSCC_data_clean.csv')
print(df.columns)


# profile = pandas_profiling.ProfileReport(df)
# profile.to_file("output.html")



df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
# df['TumourStatus'] = df['TumourStatus'].replace({'Primary': 0, 'Recurrence': 1})



# Bar plot of the number of patients by sex
sns.countplot(x="Sex", data=df)
plt.title("Number of patients by sex")
plt.show()



# Create a figure with one subplot
fig, ax = plt.subplots()
sns.histplot(df["Age"], color="#8057a5", kde=True, label="Total", ax=ax)
sns.histplot(df[df["Sex"]=="M"]["Age"], color="#0099b0", kde=True, label="Male", ax=ax)
sns.histplot(df[df["Sex"]=="F"]["Age"], color="#f16c8b", kde=True, label="Female", ax=ax)
ax.legend()
plt.show()



plt.figure(figsize=(8, 6))
sns.countplot(x='Recurrence (0= No recurrence, 1= Recurrence)', data=df)
plt.xlabel('Recurrence')
plt.ylabel('Count')
plt.title('Number of Samples by Recurrence')
plt.show()



plt.figure(figsize=(8,6))
sns.countplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', data=df)
plt.xlabel('LRD')
plt.ylabel('Count')
plt.title('Number of Samples by LRD')
plt.show()



# sns.countplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', hue='Recurrence (0= No recurrence, 1= Recurrence)', data=df)
# plt.xlabel('LRD')
# plt.ylabel('Count')
# plt.show()



df_corr = df[['Sex', 'Age', 'TumourStatus',
       'Death (0=No death, 1=Death)', 'AnatomicalLoc',
       'Grade (1= Well differentiated, 2= moderately differentiated, 3= poorly differentiated)',
       'TumourDiamater (mm)', 'TumourDepth', 'ExcisionMargin (mm)',
       'Lymphovascular (0= Absent, 1= Present)',
       'Perineural  (0= Absent, 1= Present)',
       'Sentinel lymph node biopsy  (0= Not performed, 1= Performed)',
       'Immunosuppression (0= not immunosupressed, 1= immunosuppressed)',
       'LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)',
       'Recurrence (0= No recurrence, 1= Recurrence)']]
df_corr.columns = ['Sex', 'Age', 'TumourStat', 'Death', 'AnatomLoc', 'Grade', 'TumourDiam', 'TumourDepth',
                   'ExcisionMargin', 'Lymphovascular', 'Perineural', 'SentinelLymphNode', 'Immunosuppression',
                   'LRD', 'Recurrence']
# compute the correlation matrix
corr_matrix = df_corr.corr()
# plot the heatmap
sns.set(font_scale=0.8)  # adjust the font size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.8,
            annot_kws={"size": 12}, cbar_kws={"shrink": 0.8, 'label': 'Correlation Coefficient'}, square=True)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()



df = df[df['TumourDiamater (mm)'] != 'Mohs']
df['TumourDiamater (mm)'] = pd.to_numeric(df['TumourDiamater (mm)'])


colors = ['#4287f5', '#f54242']
plt.figure(figsize=(10, 8))
# Pie Chart
recurrence_counts = df['Recurrence (0= No recurrence, 1= Recurrence)'].value_counts()
plt.pie(recurrence_counts, labels=recurrence_counts.index, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white'})
plt.title('Recurrence Distribution', fontsize=16, fontweight='bold')
plt.axis('equal')  # Equal aspect ratio ensures circular pie chart
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 8))
for label, color in zip(df['Death (0=No death, 1=Death)'].unique(), colors):
    subset = df[df['Death (0=No death, 1=Death)'] == label]
    plt.scatter(subset['TumourDiamater (mm)'], subset['TumourDepth'], color=color, alpha=0.7, label=label)
plt.title('Tumor Diameter vs. Tumor Depth', fontsize=16, fontweight='bold')
plt.xlabel('Tumor Diameter (mm)', fontsize=14)
plt.ylabel('Tumor Depth', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



print(df['TumourDiamater (mm)'].unique())
print(df['AnatomicalLoc'].unique())
plt.figure(figsize=(10, 6))
sns.boxplot(x='AnatomicalLoc', y='TumourDiamater (mm)', data=df)
plt.title('Distribution of Tumor Diameter by Anatomical Location')
plt.xlabel('Anatomical Location')
plt.ylabel('Tumor Diameter (mm)')
plt.show()



plt.boxplot([df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==1]['TumourDiamater (mm)'],
            df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==2]['TumourDiamater (mm)'],
            df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)']==3]['TumourDiamater (mm)']])
plt.xticks([1, 2, 3], ['Local', 'Regional', 'Distant'])
plt.xlabel('LRD')
plt.ylabel('Tumour Diamter (mm)')
plt.show()



sns.violinplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', y='TumourDiamater (mm)', data=df)
plt.xlabel('LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)')
plt.ylabel('Tumor Diameter (mm)')
plt.show()



sns.violinplot(x='Recurrence (0= No recurrence, 1= Recurrence)', y='TumourDepth', data=df)
plt.ylabel('Tumour Depth')
plt.show()



plt.bar(['No Recurrence', 'Recurrence'], [df[df['Recurrence (0= No recurrence, 1= Recurrence)']==0]['TumourDepth'].mean(),
                                          df[df['Recurrence (0= No recurrence, 1= Recurrence)']==1]['TumourDepth'].mean()])
plt.ylabel('Mean Tumour Depth')
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
