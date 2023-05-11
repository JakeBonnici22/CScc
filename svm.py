import pre_ml
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

X_train, X_test, y_train, y_test = train_test_split(pre_ml.scaled_data, pre_ml.target_data, test_size=0.2, random_state=42)


algorithms = [
    {
        'name': 'SVM',
        'model': SVC(),
        'param_grid': {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [0.01, 0.1, 1],
            'kernel': ['linear', 'rbf']
        }
    },
    {
        'name': 'Random Forest',
        'model': RandomForestClassifier(),
        'param_grid': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'name': 'Logistic Regression',
        'model': LogisticRegression(),
        'param_grid': {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },
    {
        'name': 'K-Nearest Neighbors',
        'model': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [2, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree']
        }
    }
]


results_df = pd.DataFrame(columns=['Algorithm', 'Parameters', 'Classification Report'])

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
    joblib.dump(best_model, f'{model_name}_best_model.pkl')

    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred)

    results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': grid_search.best_params_,
        'Best Score': grid_search.best_score_,
        'Best Model': grid_search.best_estimator_,
        'Classification Report': report
    }, ignore_index=True)

results_df.to_csv('models/model_results.csv', index=False)




# # Define the parameter grid for grid search
# param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100, 1000],
#     'gamma': [0.001, 0.01, 0.1, 1, 10],
#     'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
#     'degree': [2, 3, 4]
# }
#
# # Create the SVM model
# model = SVC()
#
# # Create the grid search object with cross validation
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(n_splits=5), verbose=3)
#
#
# # Fit the grid search on the data
# grid_search.fit(X_train, y_train)
#
# # Get the best parameters and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# best_model = grid_search.best_estimator_
# print("Best Score:", best_score)
# print("Best Parameters:", best_params)
# print("Best model:", best_model)
#
# y_true = y_test
# y_pred = best_model.predict(X_test)
#
# report = classification_report(y_test, y_pred)
# print(report)

