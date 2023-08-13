import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pandas_profiling
import pre_ml1
df = pd.read_csv(r'cSCC_data_clean.csv')


profile = pandas_profiling.ProfileReport(df)
profile.to_file("output.html")


df["TumourStatus"] = df["TumourStatus"].str.strip()
df["AnatomicalLoc"] = df["AnatomicalLoc"].str.strip()
df['TumourStatus'] = df['TumourStatus'].replace({'Primary': 0, 'Recurrence': 1})

# Recurrence by Tumour Status
sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.2)
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x="Sex", data=df, ax=ax)
ax.set_title("Number of patients by sex", fontsize=16)
total = len(df["Sex"])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
ax.tick_params(axis='both', labelsize=8)
plt.savefig("graphs/sex_count.png", dpi=300)
plt.show()


age_group_counts = pre_ml1.df['Age Group (16-78, 78-87, 87-112)'].value_counts()
colors = ['#7bccc4', '#43a2ca', '#0868ac']
# Age Group Count
sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.2)
fig, ax = plt.subplots(figsize=(10, 6))
male_counts = pre_ml1.df[pre_ml1.df['Sex'] == 1]['Age Group (16-78, 78-87, 87-112)'].value_counts()
female_counts = pre_ml1.df[pre_ml1.df['Sex'] == 0]['Age Group (16-78, 78-87, 87-112)'].value_counts()
bar_width = 0.35
age_groups = age_group_counts
bar_positions = range(len(age_groups))
ax.bar(bar_positions, male_counts, bar_width, label='Male', color=colors[0])
ax.bar(bar_positions, female_counts, bar_width, bottom=male_counts, label='Female', color=colors[1])
plt.xlabel('Age Group', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.title('Age Group Count by Gender', fontsize=20, fontweight='bold')
plt.legend(fontsize=16)
plt.xticks(bar_positions, ('16-78', '78-87', '87-112'), fontsize=18)
plt.tight_layout()
plt.savefig("graphs/age_groups_with_gender.png", dpi=300)
plt.show()

print('age group counts: ', age_group_counts)
print('male_counts: ', male_counts)
print('female_counts: ', female_counts)

std_colors = ['#7B008B', '#7B0095', '#7B009F']
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
sns.histplot(df["Age"], color=colors[2], kde=True, label="Total", palette=colors)
sns.histplot(df[df["Sex"] == 1]["Age"], color=colors[0], kde=True, label="Male", palette=colors)
sns.histplot(df[df["Sex"] == 0]["Age"], color=colors[1], kde=True, label="Female", palette=colors)
mean = df["Age"].mean()
std = df["Age"].std()

# Print data for interpretation
total_count = len(df["Age"])
male_count = len(df[df["Sex"] == 1]["Age"])
female_count = len(df[df["Sex"] == 0]["Age"])

print("Total count of age values:", total_count)
print("Count of age values for males:", male_count)
print("Count of age values for females:", female_count)
print("Mean age of the dataset:", mean)
print("Standard deviation of ages in the dataset:", std)

# Plot mean and standard deviation lines
for i in range(1, 4):
    plt.axvline(mean + i * std, linestyle='--', linewidth=1)
    plt.text(mean + i * std, plt.gca().get_ylim()[1] * 0.95, f'{i} STD', ha='right', va='top', fontsize=16,
             fontweight='bold')
    plt.text(mean + i * std, plt.gca().get_ylim()[1] * 0.9, f'{mean + i * std:.1f}', ha='right', va='top', fontsize=14)
    plt.axvline(mean - i * std, linestyle='--', linewidth=1)
    plt.text(mean - i * std, plt.gca().get_ylim()[1] * 0.95, f'-{i} STD', ha='right', va='top', fontsize=16,
             fontweight='bold')
    plt.text(mean - i * std, plt.gca().get_ylim()[1] * 0.9, f'{mean - i * std:.1f}', ha='right', va='top', fontsize=14)

plt.legend(loc='upper left', fontsize=16)
plt.tick_params(axis='both', labelsize=18)
plt.xlabel("Age", fontsize=18)
plt.ylabel("Count", fontsize=18)
plt.title("Age Distribution by Gender", fontsize=20, fontweight='bold', loc='center', pad=0)
plt.tight_layout()
plt.savefig("graphs/sex_dist.png", dpi=300)
plt.show()

# Age Group Count
sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.2)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='Recurrence (0= No recurrence, 1= Recurrence)', data=df, palette=colors)
total = len(df['Recurrence (0= No recurrence, 1= Recurrence)'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.xlabel('Recurrence', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Number of Samples by Recurrence', fontsize=16)
ax.tick_params(axis='both', labelsize=8)
plt.tight_layout()
plt.savefig("graphs/recurrence_dist.png", dpi=300)
plt.show()

sns.set(style="darkgrid")
sns.set_context("paper", font_scale=1.2)
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', data=df)
total = len(df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)'])
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', va='bottom')
plt.xlabel('LRD', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Number of Samples by LRD', fontsize=16)
plt.savefig("graphs/recurrence_dist.png", dpi=300)
plt.show()


df_corr = df[['Sex', 'Age', 'TumourStatus',
       'Death (0=No death, 1=Death)', 'AnatomicalLoc',
       'Grade (0=Well/moderately differentiated, 1=poorly differentiated)',
       'TumourDiamater (mm)', 'TumourDepth', 'ExcisionMargin (mm)',
       'Lymphovascular (0= Absent, 1= Present)',
       'Perineural  (0= Absent, 1= Present)',
       'Immunosuppression (0= not immunosupressed, 1= immunosuppressed)',
       'LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)',
       'Recurrence (0= No recurrence, 1= Recurrence)']]
df_corr.columns = ['Sex', 'Age', 'TumourStat', 'Death', 'AnatomLoc', 'Grade', 'TumourDiam', 'TumourDepth',
                   'ExcisionMargin', 'Lymphovascular', 'Perineural', 'Immunosuppression',
                   'LRD', 'Recurrence']

# Correlation Matrix
corr_matrix = df_corr.corr()
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8, 'label': 'Correlation Coefficient'}, square=True)
plt.xticks(rotation=45, ha='right', fontsize=14)
plt.yticks(rotation=0, fontsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.tight_layout()
plt.savefig("graphs/correlation_matrix.png", dpi=300)
plt.show()



df = df[df['TumourDiamater (mm)'] != 'Mohs']
df['TumourDiamater (mm)'] = pd.to_numeric(df['TumourDiamater (mm)'])


colors = ['#4287f5', '#f54242']

# Tumor Diameter vs. Tumor Depth
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
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
plt.savefig("graphs/scatter_plot.png", dpi=300)
plt.show()


# Tumor Diameter by Anatomical Location
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='AnatomicalLoc', y='TumourDiamater (mm)', data=df)
plt.title('Distribution of Tumor Diameter by Anatomical Location', fontsize=16, fontweight='bold')
plt.xlabel('Anatomical Location', fontsize=14)
plt.ylabel('Tumor Diameter (mm)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("graphs/box_plot.png", dpi=300)
plt.show()

# Tumor Diameter by LRD
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
data = [df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)'] == 1]['TumourDiamater (mm)'],
        df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)'] == 2]['TumourDiamater (mm)'],
        df[df['LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)'] == 3]['TumourDiamater (mm)']]
plt.boxplot(data)
plt.xticks([1, 2, 3], ['Local', 'Regional', 'Distant'], fontsize=12)
plt.xlabel('LRD', fontsize=14)
plt.ylabel('Tumour Diameter (mm)', fontsize=14)
plt.title('Distribution of Tumour Diameter by LRD', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/box_plot.png", dpi=300)
plt.show()


sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
sns.violinplot(x='LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', y='TumourDiamater (mm)', data=df)
plt.xlabel('LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)', fontsize=14)
plt.ylabel('Tumor Diameter (mm)', fontsize=14)
plt.title('Distribution of Tumor Diameter by LRD', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/violin_plot.png", dpi=300)
plt.show()

# Distribution of Tumour Depth by Recurrence
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
sns.violinplot(x='Recurrence (0= No recurrence, 1= Recurrence)', y='TumourDepth', data=df)
plt.ylabel('Tumour Depth', fontsize=14)
plt.title('Distribution of Tumour Depth by Recurrence', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/violin_plot_recurrence.png", dpi=300)
plt.show()

# Mean Tumour Depth by Recurrence
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
mean_depths = [df[df['Recurrence (0= No recurrence, 1= Recurrence)']==0]['TumourDepth'].mean(),
               df[df['Recurrence (0= No recurrence, 1= Recurrence)']==1]['TumourDepth'].mean()]
plt.bar(['No Recurrence', 'Recurrence'], mean_depths)
plt.ylabel('Mean Tumour Depth', fontsize=14)
plt.title('Mean Tumour Depth by Recurrence', fontsize=16, fontweight='bold')
for i, mean_depth in enumerate(mean_depths):
    plt.text(i, mean_depth, f"{mean_depth:.2f}", ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.savefig("graphs/mean_tumour_depth.png", dpi=300)
plt.show()

# Count of Tumour Grade by Recurrence
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
sns.countplot(x='Grade (0=Well/moderately differentiated, 1=poorly differentiated)', hue='Recurrence (0= No recurrence, 1= Recurrence)', data=df)
plt.xlabel('Tumour Grade', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.title('Count of Tumour Grade by Recurrence', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/tumour_grade.png", dpi=300)
plt.show()

# Recurrence by Gender
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
colors = ['#5975a4', '#cc8963']
recurrence_by_gender = df.groupby(['Sex', 'Recurrence (0= No recurrence, 1= Recurrence)']).size().unstack()
recurrence_by_gender_percent = recurrence_by_gender.apply(lambda x: x / x.sum() * 100, axis=1)
recurrence_by_gender.plot(kind='bar', stacked=True, color=colors)
total_counts = recurrence_by_gender.sum(axis=1)
for i, (index, row) in enumerate(recurrence_by_gender.iterrows()):
    for j, val in enumerate(row):
        percentage = recurrence_by_gender_percent.loc[index].iloc[j]
        plt.text(i, row[:j + 1].sum() - val / 2, f"{percentage:.1f}%", ha='center', va='center', color='white')
plt.ylabel('Count', fontsize=14)
plt.title('Recurrence by Gender', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.legend(['No Recurrence', 'Recurrence'], loc='upper left')
plt.savefig("graphs/recurrence_by_gender.png", dpi=300)
plt.show()

# Recurrence by Age
sns.set_context("paper", font_scale=1.2)
sns.set_style("darkgrid")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Recurrence (0= No recurrence, 1= Recurrence)', y='Age', data=df)
plt.ylabel('Age', fontsize=14)
plt.title('Age Distribution by Recurrence', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("graphs/age_box_plot.png", dpi=300)
plt.show()
