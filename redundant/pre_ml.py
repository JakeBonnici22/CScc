import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore", category=DeprecationWarning)


df = pd.read_csv(r'cSCC_data_clean.csv')


# Scaling numerical columns
columns_to_scale = ['TumourDiamater (mm)', 'ExcisionMargin (mm)', 'TumourDepth']
selected_data = df[columns_to_scale]
scaler = StandardScaler() #Inside CV (after train test split)
scaled_data = scaler.fit_transform(selected_data)
df[columns_to_scale] = scaled_data

# One-hot encoding categorical columns
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(df[['TumourStatus', 'AnatomicalLoc', 'HistDiag']])
encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names())
df = pd.concat([df.drop(['TumourStatus', 'AnatomicalLoc', 'HistDiag', 'Sex'], axis=1), encoded_df], axis=1)

# Display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Removing selected columns and assigning scaled_data and target_data
selected_columns = ['HospitalN', 'LRD (1= local recurrence, 2= regional recurrence, 3= distant recurrence)',
                    'Recurrence (0= No recurrence, 1= Recurrence)', 'Death (0=No death, 1=Death)', 'x0_Recurrence',
                    'x0_Primary', 'x1_cheek', 'x1_chest', 'x1_ear', 'x1_eyelid', 'x1_forehead', 'x1_jaw', 'x1_lip',
                    'x1_lower limb', 'x1_neck', 'x1_nose', 'x1_scalp', 'x1_temple', 'x1_trunk', 'x1_upper limb',
       'x2_squamous cell carcinoma']

scaled_data = df.drop(selected_columns, axis=1)

target_data = df['Recurrence (0= No recurrence, 1= Recurrence)']


print(scaled_data.columns)

