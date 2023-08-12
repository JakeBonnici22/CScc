import libs
import ml_algos
import pre_ml1
import imblearn.pipeline as imbpipeline
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV


results_df = libs.pd.DataFrame(
    columns=['Algorithm', 'Parameters', 'Best Score', 'Best Model', 'Accuracy', 'Precision', 'Recall',
             'F1-Score', 'AUC-ROC', 'AUC-PR', 'Classification Report'])

individual_models = []

for algo in ml_algos.algorithms:
    pipeline = imbpipeline.Pipeline([
        # ('preprocessor', pre_ml1.preprocessor),
        ('oversampler', pre_ml1.sampler),
        ('selector', algo['selector']),
        ('model', algo['model'])
    ])

    param_distributions = algo['param_grid']
    n_iter = 10  # Number of random parameter samples

    random_search = RandomizedSearchCV(pipeline, param_distributions=param_distributions, n_iter=n_iter,
                                       cv=libs.StratifiedKFold(n_splits=5), scoring='roc_auc', verbose=3)
    random_search.fit(pre_ml1.X_train, pre_ml1.y_train)

    best_model = random_search.best_estimator_
    model_name = algo['name']
    model_filename = f'ensemble/{model_name}_best_model.pkl'
    libs.joblib.dump(best_model, model_filename)

    y_pred = best_model.predict(pre_ml1.X_test)
    report = libs.classification_report(pre_ml1.y_test, y_pred)
    accuracy = libs.accuracy_score(pre_ml1.y_test, y_pred)
    precision = libs.precision_score(pre_ml1.y_test, y_pred)
    recall = libs.recall_score(pre_ml1.y_test, y_pred)
    f1 = libs.f1_score(pre_ml1.y_test, y_pred)
    auc_roc = libs.roc_auc_score(pre_ml1.y_test, y_pred)
    auc_pr = libs.average_precision_score(pre_ml1.y_test, y_pred)

    report_filename = f'ensemble/reports/{model_name}_classification_report.txt'
    with open(report_filename, 'w') as file:
        file.write(report)

        results_df = results_df.append({
        'Algorithm': model_name,
        'Parameters': random_search.best_params_,
        'Best Score': random_search.best_score_,
        'Best Model': model_filename,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc_roc,
        'AUC-PR': auc_pr,
        'Classification Report': report_filename
    }, ignore_index=True)

    individual_models.append((model_name, best_model))

    print(f"Predictions for {model_name}:")
    print(y_pred)
    print(f"True labels for {model_name}:")
    print(pre_ml1.y_test)
    print(f"Classification report for {model_name}_smote:")
    print(report)

if not libs.os.path.exists('ensemble/reports'):
    libs.os.makedirs('ensemble/reports')

ensemble = VotingClassifier(estimators=individual_models, voting='soft')
ensemble.fit(pre_ml1.X_train, pre_ml1.y_train)
y_pred_ensemble = ensemble.predict(pre_ml1.X_test)

ensemble_accuracy = libs.accuracy_score(pre_ml1.y_test, y_pred_ensemble)
ensemble_precision = libs.precision_score(pre_ml1.y_test, y_pred_ensemble)
ensemble_recall = libs.recall_score(pre_ml1.y_test, y_pred_ensemble)
ensemble_f1 = libs.f1_score(pre_ml1.y_test, y_pred_ensemble)
ensemble_auc_roc = libs.roc_auc_score(pre_ml1.y_test, y_pred_ensemble)
ensemble_auc_pr = libs.average_precision_score(pre_ml1.y_test, y_pred_ensemble)

ensemble_report_filename = 'ensemble/reports/ensemble_classification_report.txt'
ensemble_report = libs.classification_report(pre_ml1.y_test, y_pred_ensemble)

with open(ensemble_report_filename, 'w') as file:
    file.write(ensemble_report)

results_df = results_df.append({
    'Algorithm': 'Ensemble',
    'Parameters': None,
    'Best Score': None,
    'Best Model': None,
    'Accuracy': ensemble_accuracy,
    'Precision': ensemble_precision,
    'Recall': ensemble_recall,
    'F1-Score': ensemble_f1,
    'AUC-ROC': ensemble_auc_roc,
    'AUC-PR': ensemble_auc_pr,
    'Classification Report': ensemble_report_filename
}, ignore_index=True)

results_df.to_csv('ensemble/model_results.csv', index=False)
