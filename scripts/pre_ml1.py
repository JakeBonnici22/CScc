import libs
libs.warnings.filterwarnings('ignore')
import numpy as np
df = libs.pd.read_csv(r'cSCC_data_clean.csv')


percentile_ranges = [0, 33.33, 66.67, 100]
age_ranges = np.percentile(df['Age'], percentile_ranges)
age_labels = [1, 2, 3]  # Numerical labels
df['Age Group (' + str(int(age_ranges[0])) + '-' + str(int(age_ranges[1])) + ', ' +
                str(int(age_ranges[1])) + '-' + str(int(age_ranges[2])) + ', ' +
                str(int(age_ranges[2])) + '-' + str(int(age_ranges[3])) + ')'] = libs.pd.cut(df['Age'], bins=age_ranges,
                                                                                        labels=age_labels, right=False)


# Tumour Size Category
size_ranges = [0, 10, 20, float('inf')]
size_labels = [1, 2, 3]

df['Tumour Size Category (1=Small, 2=Medium, 3=Large)'] = libs.pd.cut(df['TumourDiamater (mm)'], bins=size_ranges,
                                                                      labels=size_labels)
# Tumour Depth Category
depth_ranges = [0, 5, float('inf')]
depth_labels = [1, 2]
df['Tumour Depth Category (1=Superficial, 2=Deep)'] = libs.pd.cut(df['TumourDepth'], bins=depth_ranges,
                                                                  labels=depth_labels)
# Excision Margin Category
margin_ranges = [0, 1, 5, float('inf')]
margin_labels = [1, 2, 3]

df['Excision Margin Category (1=<1mm, 2=1-5mm, 3=>5mm)'] = libs.pd.cut(df['ExcisionMargin (mm)'], bins=margin_ranges,
                                                                  labels=margin_labels)


cols_to_drop = ['TumourStatus', 'HospitalN', 'HistDiag', 'Recurrence (0= No recurrence, 1= Recurrence)',
                'Death (0=No death, 1=Death)', 'LRD (1= local recurrence, 2= regional recurrence,'' 3= distant recurrence)',
                'Age', 'TumourDiamater (mm)', 'TumourDepth', 'ExcisionMargin (mm)', 'AnatomicalLoc']


X_train, X_test, y_train, y_test = libs.train_test_split(df.drop(cols_to_drop, axis=1),
                                                    df['Recurrence (0= No recurrence, 1= Recurrence)'],
                                                    test_size=0.2, random_state=42,
                                                    stratify=df['Recurrence (0= No recurrence, 1= Recurrence)'])


# Outlier detection (In this particular case outliers are not removed)
# # print(y_train.value_counts())
# float_columns = ['TumourDiamater (mm)', 'TumourDepth', 'ExcisionMargin (mm)']
# X_train[float_columns] = X_train[float_columns].apply(zscore)
# threshold = 3
#
# outlier_indices = []
# for column in float_columns:
#     z_scores = X_train[column]
#     outlier_indices.extend(X_train.index[np.abs(z_scores) > threshold])
#
#
# outlier_indices = set(outlier_indices)
# valid_outlier_indices = list(outlier_indices.intersection(X_train.index))
# num_outliers = len(valid_outlier_indices)
# percentage_outliers = (num_outliers / len(X_train)) * 100
# X_train = X_train.drop(valid_outlier_indices)
# y_train = y_train.drop(valid_outlier_indices)

# print("Number of outliers removed:", num_outliers)
# print("Percentage of outliers removed:", percentage_outliers)
# num_rows_kept = len(X_train)
# print("Number of rows kept in the dataset:", num_rows_kept)
#
# print(y_train.value_counts())

# columns_to_scale = ['TumourDiamater (mm)', 'ExcisionMargin (mm)', 'TumourDepth']
# columns_to_encode = ['AnatomicalLoc']

# preprocessor = libs.ColumnTransformer([
#     # ('scaler', libs.StandardScaler(), columns_to_scale),
#     # ('encoder', libs.OneHotEncoder(handle_unknown='ignore', dtype=libs.np.int64), columns_to_encode)
# ], remainder='passthrough')

# oversampler = libs.SMOTE(random_state=42)
# sampler = libs.SMOTETomek(random_state=42)
sampler = libs.ADASYN(random_state=42)

print("Preprocessing done")