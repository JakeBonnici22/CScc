import pre_ml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score,\
    roc_auc_score, average_precision_score
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = train_test_split(pre_ml.scaled_data, pre_ml.target_data, test_size=0.2,
                                                    random_state=42)

algorithms = [
    {
        'name': 'XGBoost',
        'model': XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'learning_rate': [0.1, 0.01, 0.001]
        }
    },

    {
        'name': 'SVM',
        'model': SVC(class_weight='balanced'),
        'param_grid': {
            'C': [0.1, 1, 10],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf']

        }
    },

    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(class_weight='balanced'),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    },

    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(class_weight='balanced'),
        'param_grid': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': [None, 0.25, 0.5, 0.75]
        }
    },
    {
        'name': 'K-Nearest Neighbors',
        'model': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'leaf_size': [10, 30, 50]
        }
    },

    {
        'name': 'Decision Tree',
        'model': DecisionTreeClassifier(class_weight='balanced'),
        'param_grid': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    }
]


results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', 'Best Score', 'Best Model', 'Accuracy', 'Precision',
                                   'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR', 'Classification Report'])


for algo in algorithms:
    model_name = algo['name']
    model = algo['model']
    param_grid = algo['param_grid']

    # Perform grid search for the current algorithm
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(n_splits=5), verbose=3)
    grid_search.fit(X_train, y_train)

    # Save the best model
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    model_filename = f'models/{model_name}_best_model.pkl'
    joblib.dump(best_model, model_filename)

    # Save the model weights/parameters
    model_weights = '\n'.join([f"{key}: {value}" for key, value in best_model.get_params().items()])

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_pred)

    # Save the classification report as a separate text file
    report_filename = f'models/{model_name}_classification_report.txt'
    with open(report_filename, 'w') as file:
        file.write(report)

    # Print the predictions and true labels
    print(f"Predictions for {model_name}:")
    print(y_pred)
    print(f"True labels for {model_name}:")
    print(y_test)

    # Print the classification report
    print(f"Classification report for {model_name}:")
    print(report)


    results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': grid_search.best_params_,
        'Best Score': grid_search.best_score_,
        'Best Model': grid_search.best_estimator_,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Classification Report': report_filename,
    }, ignore_index=True)


    results_df.to_csv('models/model_results.csv', index=False)


