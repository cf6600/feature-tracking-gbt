## Refer to: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
## To understand sklearn's underlying tree structure

from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np


class CustomGBR(GradientBoostingRegressor):
    def __init__(self, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_depth=3, min_impurity_decrease=0., 
                 init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, 
                 validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0):
        super().__init__(
        loss=loss,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        criterion=criterion,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_depth=max_depth,
        init=init,
        subsample=subsample,
        max_features=max_features,
        min_impurity_decrease=min_impurity_decrease,
        random_state=random_state,
        alpha=alpha,
        verbose=verbose,
        max_leaf_nodes=max_leaf_nodes,
        warm_start=warm_start,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        tol=tol,
        ccp_alpha=ccp_alpha)
        self.iter_feature_importances_ = []
        self.detailed_impact_ = []

    def compute_weighted_feature_importances(self, tree):
        """
        Computes detailed node information for a decision tree and returns it as a DataFrame.
        Unused, we will use the already implemented on in sklearn's base tree class instead.
        """
        n_features = tree.n_features
        feature_names = ['Feature_' + str(i) for i in range(n_features)]

        node_features = []
        weighted_impurity_decreases = []
        samples_counts = []

        def recurse(node_id):
            if tree.feature[node_id] == -2:  # Check if leaf node
                return

            # Calculate the decrease in impurity weighted by the number of samples at the node
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            current_impurity = tree.impurity[node_id] * tree.n_node_samples[node_id]
            left_impurity = 0 if left_child == -2 else tree.impurity[left_child] * tree.n_node_samples[left_child]
            right_impurity = 0 if right_child == -2 else tree.impurity[right_child] * tree.n_node_samples[right_child]
            weighted_impurity_decrease = current_impurity - (left_impurity + right_impurity)

            # Collect information for the current node
            node_features.append(tree.feature[node_id])
            weighted_impurity_decreases.append(weighted_impurity_decrease)
            samples_counts.append(tree.n_node_samples[node_id])

            # Recurse for children
            if left_child != -2:
                recurse(left_child)
            if right_child != -2:
                recurse(right_child)

        # Start recursion from root node
        recurse(0)

        df = pd.DataFrame({
            'feature': node_features,
            'importance': weighted_impurity_decreases,
            'samples': samples_counts
        })
        return df

    def fit(self, X, y):
        self.iter_feature_importances_ = []

        # Call GBR's fit
        super().fit(X, y)
        
        # Save feature importances after each estimator
        for estimator in self.estimators_:
            stage_importances = estimator[0].tree_.compute_feature_importances(normalize=True)
            self.iter_feature_importances_.append(stage_importances)
        
        self.iter_feature_impacts_ = self.aggregate_feature_impacts(directional=False, normalized=True)

        return self
    

    def get_feature_importance_per_iteration(self):
        # Each row corresponds to an iteration, each column to a feature
        importances_df = pd.DataFrame(self.iter_feature_importances_)
        return importances_df
    

    def get_feature_impacts_per_iteration(self, directional=False, normalized=True):
        return self.aggregate_feature_impacts(directional, normalized)
    
    def get_feature_gradients_per_iteration(self, X, directional=False):
        return self.aggregate_feature_gradients(X, directional)


    def aggregate_feature_impacts(self, directional, normalized):
        feature_impacts = []
        for iter, estimator in enumerate(self.estimators_):
            for split in self._extract_splits(estimator[0].tree_, iteration=iter + 1, directional=directional):
                feature_impacts.append(split)
        
        df = pd.DataFrame(feature_impacts, columns=['iteration', 'feature', 'threshold', 'impact', 'samples'])
        if normalized:
            mean_impacts = df.groupby(['feature', 'iteration'])['impact'].mean().reset_index()
            impact_sums = mean_impacts.groupby('iteration')['impact'].transform('sum')
            mean_impacts['normalized_impact'] = mean_impacts['impact'] / impact_sums
            mean_impacts['samples'] = df['samples']
            df = mean_impacts
        return df.sort_values(by='iteration')
    

    def _extract_splits(self, tree, node_id=0, iteration=0, path=[], directional=False):
        # Check for leaf nodes
        if tree.children_left[node_id] == tree.children_right[node_id] == -1:
            return []

        # Check node's split details; not a leaf node
        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child_node_id, right_child_node_id = tree.children_left[node_id], tree.children_right[node_id]
        
        # Calculate the impact of a feature split, defined as value of left child - value of right child
        if not directional:
            impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum())
        else:
            impact = tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()


        # Get sample sizes of the splits
        left_child_sample_size, right_child_sample_size = tree.n_node_samples[left_child_node_id], tree.n_node_samples[right_child_node_id]

        splits_info = [{
            'iteration': iteration,
            'feature': feature_index,
            'threshold': threshold,
            'impact': impact, # Represents the magnitude of change from low to high values of a particular feature in that iteration
            'samples': left_child_sample_size + right_child_sample_size
        }]

        # Recursively get split information for the left sub-nodes
        if tree.children_left[node_id] != -1:
            left_path = path + [f'{feature_index} <= {threshold}']
            splits_info += self._extract_splits(tree, tree.children_left[node_id], iteration, left_path)

        # Recursively get split information for the right sub-nodes
        if tree.children_right[node_id] != -1:
            right_path = path + [f'{feature_index} > {threshold}']
            splits_info += self._extract_splits(tree, tree.children_right[node_id], iteration, right_path)

        return splits_info
    
    def aggregate_feature_gradients(self, X, directional):
        feature_gradients = []
        for iter, estimator in enumerate(self.estimators_):
            for split in self._extract_splits_grad(estimator[0].tree_, X, iteration=iter + 1, directional=directional):
                feature_gradients.append(split)
        
        df = pd.DataFrame(feature_gradients, columns=['iteration', 'feature', 'gradient', 'impact', 'samples'])
        return df
    
    def _extract_splits_grad(self, tree, X, node_id=0, iteration=0, path=[], directional=False):
        # Check for leaf nodes
        if tree.children_left[node_id] == tree.children_right[node_id] == -1:
            return []

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child_node_id, right_child_node_id = tree.children_left[node_id], tree.children_right[node_id]

        if not directional:
            impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum())
        else:
            impact = tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()
        
        # Apply the split condition to the dataset to separate left and right samples
        left_samples = X[X[:, feature_index] <= threshold]
        right_samples = X[X[:, feature_index] > threshold]


        # Calculate the average feature value for the samples in each child node
        avg_feature_value_left = left_samples[:, feature_index].mean() if len(left_samples) > 0 else 0
        avg_feature_value_right = right_samples[:, feature_index].mean() if len(right_samples) > 0 else 0

        # Calculate the feature distance as the difference in averages
        feature_distance = avg_feature_value_right - avg_feature_value_left
        if feature_distance == 0:
            gradient = 0
        else:
            gradient = impact / feature_distance

        # Additional information to track, like in your original function
        splits_info = [{
            'iteration': iteration,
            'feature': feature_index,
            'gradient': gradient,
            'impact': impact,
            'samples': len(left_samples) + len(right_samples),
        }]

        # Recursively process child nodes, assuming self._extract_splits is correctly implemented
        if tree.children_left[node_id] != -1:
            splits_info += self._extract_splits_grad(tree, left_samples, tree.children_left[node_id], 
                                                     iteration+1, path + [f'{feature_index} <= {threshold}'])

        if tree.children_right[node_id] != -1:
            splits_info += self._extract_splits_grad(tree, right_samples, tree.children_right[node_id], 
                                                     iteration+1, path + [f'{feature_index} > {threshold}'])

        return splits_info
    
    def get_prediction_feature_impacts(self, X, directional=False):
        """
        Takes in an array of numbers corresponding to a single data point
        """
        feature_impacts = []

        for iter_num, estimator in enumerate(self.estimators_):
            tree = estimator[0].tree_
            decision_path = estimator[0].decision_path(X).toarray()[0]
            active_nodes = np.where(decision_path == 1)[0]

            for node_id in active_nodes:
                # Skip if current node is a leaf node
                if tree.children_left[node_id] == tree.children_right[node_id] == -1:
                    continue

                feature_index = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                left_child_node_id, right_child_node_id = tree.children_left[node_id], tree.children_right[node_id]
                # Calculate the impact of a feature split, defined as value of left child - value of right child
                if not directional:
                    impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum())
                else:
                    impact = tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()
                feature_impacts.append({
                    'threshold': threshold,
                    'iteration': iter_num + 1,
                    'feature': feature_index,
                    'impact': impact
                })

        df_impacts = pd.DataFrame(feature_impacts)
        return df_impacts

    def get_predictions_feature_impacts(self, X, directional=False):
        """
        Takes in mulitple data points and consolidates feature impacts on predictions
        """
        dfs = []
        for i in range(len(X)):
            dfs.append(self.get_prediction_feature_impacts(np.array(X.iloc[i, :]).reshape(1, -1), directional))
        return pd.concat(dfs)


    def get_prediction_feature_gradients(self, X, directional=False):
        """
        Takes in an array of numbers corresponding to a single data point
        """
        feature_gradients = []

        for iter_num, estimator in enumerate(self.estimators_):
            tree = estimator[0].tree_
            decision_path = estimator[0].decision_path(X).toarray()[0]
            active_nodes = np.where(decision_path == 1)[0]

            for node_id in active_nodes:
                # Skip if current node is a leaf node
                if tree.children_left[node_id] == tree.children_right[node_id] == -1:
                    continue

                feature_index = tree.feature[node_id]
                threshold = tree.threshold[node_id]
                left_child_node_id, right_child_node_id = tree.children_left[node_id], tree.children_right[node_id]
                # Calculate the impact of a feature split, defined as value of left child - value of right child
                if not directional:
                    impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum())
                else:
                    impact = tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()
                feature_gradients.append({
                    'threshold': threshold,
                    'iteration': iter_num + 1,
                    'feature': feature_index,
                    'impact': impact
                })

        df_gradients = pd.DataFrame(feature_gradients)
        return df_gradients

    def get_predictions_feature_gradients(self, X, directional=False):
        """
        Takes in mulitple data points and consolidates feature impacts on predictions
        """
        dfs = []
        for i in range(len(X)):
            dfs.append(self.get_prediction_feature_impacts(np.array(X.iloc[i, :]).reshape(1, -1), directional))
        return pd.concat(dfs)

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from sklearn.utils import check_array
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    diabetes_data = load_diabetes()
    X_data, y_data = diabetes_data.data, diabetes_data.target
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    gbr = CustomGBR(n_estimators=10, learning_rate=0.1, max_depth=3)
    gbr.fit(X_train, y_train)
    feature_gradients_per_iter = gbr.get_feature_gradients_per_iteration(X_test)
