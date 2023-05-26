import libs
libs.warnings.filterwarnings('ignore')


df = libs.pd.read_csv(r'cSCC_data_clean.csv')

# Split the data into train and test sets
X_train, X_test, y_train, y_test = libs.train_test_split(df.drop(['Recurrence (0= No recurrence, 1= Recurrence)',
                                                            'Death (0=No death, 1=Death)',
                                                            'LRD (1= local recurrence, 2= regional recurrence,'
                                                            ' 3= distant recurrence)'], axis=1),
                                                    df['Recurrence (0= No recurrence, 1= Recurrence)'],
                                                    test_size=0.2, random_state=42)


columns_to_scale = ['TumourDiamater (mm)', 'ExcisionMargin (mm)', 'TumourDepth']
columns_to_encode = ['TumourStatus', 'AnatomicalLoc', 'HistDiag']

preprocessor = libs.ColumnTransformer([
    ('scaler', libs.StandardScaler(), columns_to_scale),
    ('encoder', libs.OneHotEncoder(handle_unknown='ignore', dtype=libs.np.int64), columns_to_encode)
])


