import libs
import pre_ml1

algorithms = [

    {
        'name': 'SVM',
        'model': libs.SVC(class_weight='balanced'),
        'param_grid': {
            'model__C': [0.1, 1, 10],
            'model__gamma': [0.01, 0.1, 1],
            'model__kernel': ['linear', 'rbf']

        }
    },

    {
        'name': 'XGBoost',
        'model': libs.XGBClassifier(scale_pos_weight=(len(pre_ml1.y_train) - sum(pre_ml1.y_train)) / sum(pre_ml1.y_train)),
        'param_grid': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.1, 0.01, 0.001]
        }
    },

    {
        'name': 'Random Forest',
        'model': libs.RandomForestClassifier(class_weight='balanced'),
        'param_grid': {
            'model__n_estimators': [100, 200, 500],
            'model__max_depth': [2, 3, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__bootstrap': [True, False]
        }
    },

    {
        'name': 'Logistic Regression',
        'model': libs.LogisticRegression(class_weight='balanced'),
        'param_grid': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2'],
            'model__solver': ['liblinear', 'saga'],
            'model__l1_ratio': [None, 0.25, 0.5, 0.75]
        }
    },
    {
        'name': 'K-Nearest Neighbors',
        'model': libs.KNeighborsClassifier(),
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
        'param_grid': {
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['auto', 'sqrt', 'log2']
        }
    }
]