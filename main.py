from data.data_loader import load_all_data
from data.data_cleaning import clean_all_data
from data.feature_engineering import create_full_feature_set_dask, select_features_by_variance
from eda.exploratory_analysis import (
    plot_target_distribution, plot_missing_values, plot_feature_distributions, plot_correlation_heatmap, plot_new_feature_analysis
)
from models.model_training import split_data, train_logistic_regression
from utils.metrics import map7_from_dataframe
from utils.submission import generate_submission
import os
import pandas as pd

# All modules are now in the root directory and can be imported directly as packages

def main():
    # 1. Load data
    data = load_all_data()
    print('Data loaded.')

    # 2. Clean data
    cleaned = clean_all_data(data)
    print('Data cleaned.')

    # 3. Feature engineering: use Dask for add_event/add_trans aggregation
    dataset_dir = os.path.join(os.path.dirname(__file__), 'dataset')
    add_event_path = os.path.join(dataset_dir, 'add_event_converted.parquet')
    add_trans_path = os.path.join(dataset_dir, 'add_trans_converted.parquet')
    train = create_full_feature_set_dask(
        cleaned['train'], cleaned['offer_metadata'], add_event_path, add_trans_path
    )
    test = create_full_feature_set_dask(
        cleaned['test'], cleaned['offer_metadata'], add_event_path, add_trans_path
    )
    print('Full feature set created.')

    # 4. EDA and feature analysis
    print('Running EDA...')
    plot_target_distribution(train)
    plot_missing_values(train)
    # Example: plot distributions for a few original and new features
    plot_feature_distributions(train, features=['f1', 'f2', 'event_count_user', 'total_trans_amt_user'])
    plot_correlation_heatmap(train)
    plot_new_feature_analysis(train)
    print('EDA complete.')

    # 5. Feature selection (variance threshold, can add more methods later)
    feature_cols = train.drop(columns=['y']).columns
    selected = select_features_by_variance(train.drop(columns=['y']), threshold=0.0)
    print(f'Selected {selected.shape[1]} features with non-zero variance.')
    # Add target column back for further analysis if needed
    selected['y'] = train['y']

    # 6. Model training and MAP@7 evaluation
    print('Training baseline Logistic Regression...')
    X_train, X_val, y_train, y_val = split_data(selected, target_col='y', test_size=0.2, random_state=42)
    model = train_logistic_regression(X_train, y_train)
    print('Model trained.')

    # Predict on validation set for MAP@7
    val_df = X_val.copy()
    val_df['y'] = y_val
    val_df['pred'] = model.predict_proba(X_val)[:, 1]
    # Add id2, id3 columns from original train set (assuming index alignment)
    val_df = val_df.reset_index(drop=True)
    orig_val = train.loc[X_val.index, ['id2', 'id3']].reset_index(drop=True)
    val_df['id2'] = orig_val['id2']
    val_df['id3'] = orig_val['id3']
    map7 = map7_from_dataframe(val_df, id_col='id2', offer_col='id3', label_col='y', pred_col='pred', k=7)
    print(f'MAP@7 on validation set: {map7:.5f}')

    # 7. Generate submission for test set
    print('Generating submission file...')
    test_selected = test[selected.drop(columns=["y"]).columns]
    test_pred = model.predict_proba(test_selected)[:, 1]
    submission_df = test[['id2', 'id3']].copy()
    submission_df['pred'] = test_pred
    generate_submission(submission_df, id_col='id2', offer_col='id3', pred_col='pred', k=7, output_path='submission.csv')
    print('Pipeline complete.')

if __name__ == '__main__':
    main()
