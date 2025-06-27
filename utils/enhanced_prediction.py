# enhanced_prediction.py
def generate_advanced_predictions(trainer, test_data, selected_features):
    """Generate predictions using the trained ensemble"""
    
    # Prepare test data
    X_test, _ = trainer.prepare_data(test_data)
    X_test = X_test[selected_features]
    
    # Get ensemble predictions
    ensemble_pred, individual_preds = trainer.create_ensemble_predictions(X_test)
    
    # Get meta-learner predictions if available
    if 'meta_learner' in trainer.models:
        meta_features = np.column_stack(list(individual_preds.values()))
        meta_pred = trainer.models['meta_learner'].predict_proba(meta_features)[:, 1]
        
        # Use meta-learner as final prediction
        final_pred = meta_pred
    else:
        final_pred = ensemble_pred
    
    return final_pred

# Generate predictions
test_predictions = generate_advanced_predictions(trainer, test_engineered, selected_features)

# Create submission
submission = pd.DataFrame({
    'id1': test_engineered['id1'],
    'y': test_predictions
})

submission.to_csv('advanced_submission.csv', index=False)
print("Advanced submission saved!")
