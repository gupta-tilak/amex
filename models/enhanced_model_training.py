# enhanced_model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AdvancedModelTrainer:
    def __init__(self, selected_features):
        self.selected_features = selected_features
        self.models = {}
        self.scalers = {}
        
    def prepare_data(self, df, target_col='y'):
        """Prepare data for training"""
        X = df[self.selected_features].copy()
        y = df[target_col] if target_col in df.columns else None
        
        # Handle any remaining categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = pd.Categorical(X[col]).codes
        
        return X, y
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models for ensemble"""
        
        # Model configurations
        models_config = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }
        
        # Train models
        for name, model in models_config.items():
            print(f"Training {name}...")
            
            if name in ['xgboost', 'lightgbm']:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val, val_pred)
            ap_score = average_precision_score(y_val, val_pred)
            
            print(f"{name} - AUC: {auc_score:.4f}, AP: {ap_score:.4f}")
            
            self.models[name] = model
    
    def create_ensemble_predictions(self, X):
        """Create ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            predictions[name] = model.predict_proba(X)[:, 1]
        
        # Simple average ensemble
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions
    
    def train_meta_learner(self, X_train, y_train, X_val, y_val):
        """Train a meta-learner for stacking"""
        
        # Get base model predictions on validation set
        base_predictions_train = []
        base_predictions_val = []
        
        # Use cross-validation to get unbiased predictions for training
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            train_preds = np.zeros(len(X_train))
            
            for train_idx, val_idx in kf.split(X_train, y_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train = y_train.iloc[train_idx]
                
                # Clone and train model on fold
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                train_preds[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]
            
            base_predictions_train.append(train_preds)
            
            # Predictions on actual validation set
            val_preds = model.predict_proba(X_val)[:, 1]
            base_predictions_val.append(val_preds)
        
        # Create meta-features
        meta_X_train = np.column_stack(base_predictions_train)
        meta_X_val = np.column_stack(base_predictions_val)
        
        # Train meta-learner
        meta_learner = LogisticRegression(random_state=42)
        meta_learner.fit(meta_X_train, y_train)
        
        # Evaluate meta-learner
        meta_pred = meta_learner.predict_proba(meta_X_val)[:, 1]
        meta_auc = roc_auc_score(y_val, meta_pred)
        meta_ap = average_precision_score(y_val, meta_pred)
        
        print(f"Meta-learner - AUC: {meta_auc:.4f}, AP: {meta_ap:.4f}")
        
        self.models['meta_learner'] = meta_learner
        
        return meta_pred

def train_advanced_models(train_data, selected_features, target_col='y'):
    """Main function to train advanced models"""
    
    # Initialize trainer
    trainer = AdvancedModelTrainer(selected_features)
    
    # Prepare data
    X, y = trainer.prepare_data(train_data, target_col)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")
    
    # Train ensemble models
    trainer.train_ensemble_models(X_train, y_train, X_val, y_val)
    
    # Train meta-learner
    meta_pred = trainer.train_meta_learner(X_train, y_train, X_val, y_val)
    
    return trainer

# Usage in your pipeline
# trainer = train_advanced_models(train_engineered, selected_features)
