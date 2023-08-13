import libs
import pre_ml1
import numpy as np

cost_matrix = np.array([[0, 5],
                       [10, 0]])

class_weights = {0: 1 / cost_matrix[0, 0], 1: 1 / cost_matrix[1, 1]}
base_estimator = libs.RandomForestClassifier(class_weight='balanced')

num_features_to_remove = 5

# Algorithms to compare.
algorithms = [
    {
        'name': 'SVM',
        'model': libs.SVC(class_weight='balanced', probability=True),
        'selector': None,
        'param_grid': {
            'model__C': [0.1, 1, 10],
            'model__gamma': [0.01, 0.1, 1],
            'model__kernel': ['rbf']
        }
    }
    ,
    {
        'name': 'Voting Classifier',
        'model': libs.VotingClassifier(
            estimators=[
                ('brf', libs.BalancedRandomForestClassifier(class_weight='balanced')),
                ('ada', libs.AdaBoostClassifier(base_estimator=libs.RandomForestClassifier(class_weight='balanced'))),
                ('xgb', libs.XGBClassifier(scale_pos_weight=6)),
            ],
            voting='soft'
        ),
        'selector': libs.RFECV(estimator=libs.RandomForestClassifier(class_weight='balanced')),
        'param_grid': {
            'model__weights': ['soft', 'hard'],
            'model__voting': ['hard', 'soft']
        }
    },
    {
        'name': 'Bagging Classifier',
        'model': libs.BaggingClassifier(
            base_estimator=libs.RandomForestClassifier(class_weight='balanced'),
            n_estimators=10,
            random_state=42
        ),
        'selector': libs.RFECV(estimator=libs.RandomForestClassifier(class_weight='balanced')),
        'param_grid': {
            'model__n_estimators': [5, 10, 15],
            'model__base_estimator__max_depth': [3, 5, 7],
            'model__base_estimator__min_samples_split': [2, 4, 6]
        }
    },

    {
        'name': 'Logistic Regression',
        'model': libs.LogisticRegression(class_weight='balanced', penalty='l2', solver='liblinear'),
        'selector': libs.RFE(estimator=libs.LogisticRegression(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l1', 'l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__l1_ratio': [None, 0.25, 0.5, 0.75]
        }
    }
    ,
    {
        'name': 'XGBoost',
        'model': libs.XGBClassifier(class_weight={0: 1, 1: 5}),
        'selector': libs.RFE(estimator=libs.XGBClassifier(scale_pos_weight=10), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.1, 0.01, 0.001]
        }
    },

    {
        'name': 'Balanced Bagging Classifier',
        'model': libs.BalancedBaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=10,
            random_state=42,
            sampling_strategy='auto',
            replacement=False,
        ),
        'selector': libs.RFE(estimator=libs.RandomForestClassifier(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__n_estimators': [100, 200, 500],
            'model__base_estimator__min_samples_split': [2, 4, 6]
        }
    },
    {
        'name': 'Balanced Random Forest',
        'model': libs.BalancedRandomForestClassifier(class_weight='balanced'),
        'selector': libs.RFE(estimator=libs.RandomForestClassifier(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [2, 3, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__bootstrap': [True]
        }
    },
    {
        'name': 'ADA',
        'model': libs.AdaBoostClassifier(base_estimator=libs.RandomForestClassifier(class_weight='balanced')),
        'selector': libs.RFE(estimator=libs.RandomForestClassifier(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1]
        }
    },
    {
        'name': 'Random Forest',
        'model': libs.RandomForestClassifier(class_weight='balanced'),
        'selector': libs.RFE(estimator=libs.RandomForestClassifier(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [2, 3, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__bootstrap': [True, False]
        }
    },
    {
        'name': 'K-Nearest Neighbors',
        'model': libs.KNeighborsClassifier(),
        'selector': None,
        'param_grid': {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
            'model__algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'model__leaf_size': [10, 30, 50]
        }
    },
    {
        'name': 'Decision Tree',
        'model': libs.DecisionTreeClassifier(class_weight='balanced'),
        'selector': libs.RFE(estimator=libs.DecisionTreeClassifier(class_weight='balanced'), n_features_to_select=num_features_to_remove),
        'param_grid': {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    }
    ]

print("Algorithms defined")
