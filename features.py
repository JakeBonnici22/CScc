import libs
import pre_ml1

model_filename = 'modelswpipe/XGBoost_best_model.pkl'
pipeline = libs.joblib.load(model_filename)
xg = pipeline.named_steps['model']


selected_features_mask = pipeline.named_steps['selector'].get_support()
preprocessor = pipeline.named_steps['preprocessor']
feature_names = []

for transformer_name, transformer, features in preprocessor.transformers_:
    if transformer_name == 'scaler':
        feature_names.extend(features)
    elif transformer_name == 'encoder':
        encoded_feature_names = []
        for category, feature in zip(transformer.categories_, features):
            encoded_feature_names.extend([f"{feature}_{value}" for value in category])
        feature_names.extend(encoded_feature_names)
    else:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out())
        else:
            feature_names.extend(features)

selected_feature_names = [feature_names[i] for i, mask_value in enumerate(selected_features_mask) if mask_value]


mapping = {index: column_name for index, column_name in enumerate(pre_ml1.X_train.columns)}
feature_names_mapped = [mapping.get(feature, feature) for feature in feature_names]
selected_feature_names_mapped = [mapping.get(feature, feature) for feature in selected_feature_names]

print(selected_feature_names_mapped)
print(feature_names_mapped)









libs.xgb.plot_importance(xg, importance_type='weight')
libs.plt.title('XGBoost Feature Importance')
libs.plt.xlabel('Importance Score')
libs.plt.ylabel('Feature Name')

# Get the tick positions and labels for the x-axis
ticks, _ = libs.plt.xticks()
ticks = [int(tick) for tick in ticks]  # Convert the ticks to a list of integers

# Update the tick labels with the selected feature names
selected_ticks = [feature_names_mapped[i] for i in ticks if i < len(feature_names_mapped)]

# Set the tick positions and labels
libs.plt.xticks(ticks[:len(selected_ticks)], selected_ticks[:len(ticks)], rotation=90)

libs.plt.show()