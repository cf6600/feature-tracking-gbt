import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importances_per_iter(feature_importances_per_iter_df, feature_name_mappings=None):
    """
    Takes in a dataframe of feature importances, each index corresponding to an iteration and each column a feature
    """
    df = pd.DataFrame(feature_importances_per_iter_df)
    if feature_name_mappings:
        df = df.rename(columns=feature_name_mappings)

    # Smooth plot with moving average of past 1/10 of total iterations; minimum of 5
    window_size = len(df) // 10
    df = df.rolling(window=5 if window_size < 5 else window_size).mean()

    for feature_idx in range(len(df.columns)):
        plt.plot(df.iloc[:, feature_idx], label=df.columns[feature_idx])

    plt.xlabel('Iteration')
    plt.ylabel('Feature Importance')
    plt.title('Feature Importance Over Iterations')
    plt.legend()
    plt.show()


def plot_feature_impacts_per_iter(feature_impacts_df, weighted_by_samples=False, 
                                  feature_name_mappings=None, n_samples=1):
    """
    Takes in a dataframe of feature impacts. Must have feature, impact, and iteration column.
    If weighted_by_samples is used, dataframe must have a samples column.
    """
    df = pd.DataFrame(feature_impacts_df)
    if feature_name_mappings:
        df['feature'] = df['feature'].apply(lambda x: feature_name_mappings[x])

    # Smooth plot with moving average of past 1/10 of total iterations; minimum of 10
    n_iterations = len(df['iteration'].unique())
    window_size =  n_iterations // 10 

    plt.figure(figsize=(14, 6))
    for feature in df['feature'].unique():
        df_feature = pd.DataFrame(df[df['feature'] == feature])
        if weighted_by_samples:
            df_feature['weighted_impact'] = df_feature['impact'] * df_feature['samples'] / n_samples
            weighted_impact_per_iteration = df_feature.groupby('iteration')['weighted_impact'].sum()
            impact_per_iteration = weighted_impact_per_iteration
        else:
            impact_per_iteration = df_feature.groupby('iteration')['impact'].sum()
        impact_per_iteration = impact_per_iteration.reindex(range(1, n_iterations + 1), fill_value=0)
        impact_per_iteration_smoothed = impact_per_iteration.rolling(window=10 if window_size < 10 else window_size).sum()
        plt.plot(impact_per_iteration_smoothed.index, impact_per_iteration_smoothed, label=feature)
    plt.title('Consolidated Impact per Iteration for Each Feature')
    plt.xlabel('Iteration')
    plt.ylabel('Total Impact')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_gradients_per_iter(feature_gradients_df, weighted_by_samples=False, feature_name_mappings=None):
    """
    Takes in a dataframe of feature impacts. Must have feature, impact, and iteration column.
    If weighted_by_samples is used, dataframe must have a samples column.
    """
    df = pd.DataFrame(feature_gradients_df)
    if feature_name_mappings:
        df['feature'] = df['feature'].apply(lambda x: feature_name_mappings[x])

    # Smooth plot with moving average of past 1/10 of total iterations; minimum of 5
    n_iterations = len(df['iteration'].unique())
    window_size =  n_iterations // 10

    plt.figure(figsize=(14, 6))
    for feature in df['feature'].unique():
        df_feature = pd.DataFrame(df[df['feature'] == feature])
        if weighted_by_samples:
            df_feature['weighted_gradients'] = df_feature['gradient'] * df_feature['samples']
            weighted_impact_per_iteration = df_feature.groupby('iteration')['weighted_gradients'].sum()
            impact_per_iteration = weighted_impact_per_iteration
        else:
            impact_per_iteration = df_feature.groupby('iteration')['gradient'].sum()
        impact_per_iteration = impact_per_iteration.reindex(range(1, n_iterations + 1), fill_value=0)
        impact_per_iteration_smoothed = impact_per_iteration.rolling(window=5 if window_size < 5 else window_size).sum()
        plt.plot(impact_per_iteration_smoothed.index, impact_per_iteration_smoothed, label=feature)
    plt.title('Consolidated Gradient per Iteration for Each Feature')
    plt.xlabel('Iteration')
    plt.ylabel('Total Gradients')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_total_feature_impacts(feature_impacts_per_iter_df, weighted_by_samples=False, feature_name_mappings=None):
    df = pd.DataFrame(feature_impacts_per_iter_df)
    if feature_name_mappings:
        df['feature'] = df['feature'].apply(lambda x: feature_name_mappings[x])

    if weighted_by_samples:
        df['weighted_impact'] = df['impact'] * df['samples']
        df_weighted_grouped = df.groupby('feature')['weighted_impact'].sum()
        df_grouped = df_weighted_grouped
    else:
        df_grouped = df.groupby('feature')['impact'].sum()
    plt.figure(figsize=(10, 6))
    df_grouped.plot(kind='bar')
    plt.title('Total Impact for Each Feature')
    plt.xlabel('Feature')
    plt.ylabel('Total Impact')
    plt.xticks(rotation=45)
    plt.show()


def plot_prediction_feature_impacts(predictions_df, feature_name_mappings=None):
    df = pd.DataFrame(predictions_df)
    if feature_name_mappings:
        df['feature'] = df['feature'].apply(lambda x: feature_name_mappings[x])

    aggregated_impacts = df.groupby('feature')['impact'].sum().reset_index()
    aggregated_impacts = aggregated_impacts.sort_values(by='feature')

    aggregated_impacts.plot(x='feature', y='impact', kind='bar', figsize=(10, 6))
    plt.title('Aggregated Feature Impacts for Single Prediction')
    plt.xlabel('Feature')
    plt.ylabel('Impacts')
    plt.xticks(rotation=45)
    plt.show()


def plot_prediction_feature_gradients(predictions_df, feature_name_mappings=None):
    df = pd.DataFrame(predictions_df)
    if feature_name_mappings:
        df['feature'] = df['feature'].apply(lambda x: feature_name_mappings[x])

    aggregated_impacts = df.groupby('feature')['gradient'].sum().reset_index()
    aggregated_impacts = aggregated_impacts.sort_values(by='feature')

    aggregated_impacts.plot(x='feature', y='gradient', kind='bar', figsize=(10, 6))
    plt.title('Aggregated Feature Gradients for Predictions')
    plt.xlabel('Feature')
    plt.ylabel('Gradients')
    plt.xticks(rotation=45)
    plt.show()