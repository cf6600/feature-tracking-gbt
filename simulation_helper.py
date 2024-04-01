import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from helper_plots import plot_feature_importances_per_iter, plot_feature_impacts_per_iter, plot_feature_gradients_per_iter, plot_total_feature_impacts, plot_prediction_feature_impacts
from feature_tracker import CustomGBR
from sklearn.metrics import mean_squared_error
import optuna


def generate_dataset(seed=42):
    np.random.seed(seed)
    n_samples = 1000
    X_noisy = np.random.randn(n_samples, 14)
    # Generate 1 predictive feature - linearly related to target var but with some noise
    X_predictive = np.linspace(0, 10, n_samples) + np.random.normal(0, 1, n_samples)
    X = np.hstack((X_noisy, X_predictive.reshape(-1, 1)))
    y = X_predictive * 2 + np.random.normal(0, 1, n_samples)

    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(15)])
    feature_name_mappings = {i: f'feature_{i+1}' for i in range(15)}
    df['target'] = y

    X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], 
                                                        test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test


def objective(trial, X_train, y_train, X_test, y_test):
    # Define the hyperparameters to tune
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'random_state': 42
    }

    # Initialize and train the model
    gbr = CustomGBR(**param)
    gbr.fit(X_train, y_train)
    
    # Make predictions and evaluate the model
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def tune_hyperparameters(X_train, y_train, X_test, y_test, n_trials=20):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=n_trials)
    return study.best_params


def train_eval_gbr_es(X_train, y_train, X_test, y_test, random_state=42):
    # Train CustomGBR with optuna
    X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state)
    best_params = tune_hyperparameters(X_train_opt, y_train_opt, X_val_opt, y_val_opt, n_trials=100)
    
    # Train the model with the best hyperparameters
    print(f'Chosen best params are: {best_params}')
    best_params['validation_fraction'] = 0.2
    best_params['n_iter_no_change'] = 10
    best_params['tol'] = 0.01
    gbr = CustomGBR(**best_params)
    # gbr = CustomGBR(n_estimators=500, 
    #                 learning_rate=0.1, 
    #                 max_depth=3,
    #                 validation_fraction=0.2, 
    #                 n_iter_no_change=10, 
    #                 tol=0.01, 
    #                 random_state=random_state)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE) for normal early stopping with {gbr.n_estimators_} estimators: {mse:.5f}")
    return gbr, mse, best_params


def get_feature_importance_early_stopping_iter(feature_importances_df, feature_idx, threshold=0.1, n_iter_no_change=10):
    df_feature_per_iter = pd.DataFrame(feature_importances_df[feature_idx])
    df_feature_per_iter['rolling_max_importance'] = df_feature_per_iter[feature_idx].rolling(n_iter_no_change).max()
    df_below_thresh = df_feature_per_iter[df_feature_per_iter['rolling_max_importance'] < threshold]
    if len(df_below_thresh) < 1:
        return len(df_feature_per_iter)
    return min(df_below_thresh.index)


def get_feature_impact_early_stopping_iter(feature_impacts_df, feature_idx, threshold=0.1, n_iter_no_change=2):
    """
    Feature impacts here only works for non-directional impacts.
    """
    df = pd.DataFrame(feature_impacts_df[feature_impacts_df['feature'] == feature_idx])
    df_feature_per_iter = df.groupby('iteration').sum()
    df_feature_per_iter['rolling_max_impact'] = df_feature_per_iter['normalized_impact'].rolling(n_iter_no_change).max()
    df_below_thresh = df_feature_per_iter[df_feature_per_iter['rolling_max_impact'] < threshold]
    if len(df_below_thresh) < 1:
        return len(df_feature_per_iter)
    return min(df_below_thresh.index)


def train_eval_gbr_feature_es(gbr_trained, best_params, X_train, y_train, X_test, y_test, feature_stopping_type='importance'):
    if feature_stopping_type == 'impact':
        feature_impacts_per_iter = gbr_trained.get_feature_impacts_per_iteration()
        plot_feature_impacts_per_iter(feature_impacts_per_iter)
        n_iter_feature_es = get_feature_impact_early_stopping_iter(feature_impacts_per_iter, 14, threshold=0.25, n_iter_no_change=3)
    else:
        feature_importances_per_iter = gbr_trained.get_feature_importance_per_iteration()
        plot_feature_importances_per_iter(feature_importances_per_iter)
        n_iter_feature_es = get_feature_importance_early_stopping_iter(feature_importances_per_iter, 14, threshold=0.25, n_iter_no_change=3)
    best_params['n_estimators'] = n_iter_feature_es
    gbr = CustomGBR(**best_params)
    gbr.fit(X_train, y_train)
    y_pred = gbr.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f'Mean Squared Error (MSE) for feature importance early',
          f'stopping with with {gbr.n_estimators_} estimators: {mse:.5f}')
    return n_iter_feature_es, mse


def train_eval_simulation(seed, feature_stopping_type='importance'):
    X_train, X_test, y_train, y_test = generate_dataset(seed)
    gbr_trained, mse_es, best_params = train_eval_gbr_es(X_train, y_train, X_test, y_test, random_state=seed)
    n_iter_es = gbr_trained.n_estimators_
    n_iter_feature_es, mse_feature_es = train_eval_gbr_feature_es(gbr_trained, best_params, X_train, y_train, 
                                               X_test, y_test, feature_stopping_type)
    return {'seed': seed, 'n_iter_es': n_iter_es, 'mse_es': mse_es, 
            'n_iter_feature_es': n_iter_feature_es, 'mse_feature_es': mse_feature_es}


def train_eval_simulations(n_seeds, feature_stopping_type='importance'):
    results = []
    for seed in range(n_seeds):
        results.append(train_eval_simulation(seed, feature_stopping_type))
        print(f'Seed {seed} complete.')
    df_results = pd.DataFrame(results)
    df_results['feature_es_improvement'] = df_results['mse_es'] >= df_results['mse_feature_es']
    return df_results

    