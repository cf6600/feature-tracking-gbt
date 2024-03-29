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

    def fit(self, X, y):
        self.iter_feature_importances_ = []

        # Call GBR's fit
        super().fit(X, y)
        
        # Save feature importances after each estimator
        for estimator in self.estimators_:
            stage_importances = estimator[0].tree_.compute_feature_importances(normalize=False)
            self.iter_feature_importances_.append(stage_importances)
        
        self.iter_feature_impacts_ = self.aggregate_feature_impacts()

        return self
    

    def get_feature_importance_per_iteration(self):
        # Each row corresponds to an iteration, each column to a feature
        importances_df = pd.DataFrame(self.iter_feature_importances_)
        return importances_df
    

    def get_feature_impacts_per_iteration(self):
        return self.aggregate_feature_impacts()


    def aggregate_feature_impacts(self):
        feature_impacts = []
        for iter, estimator in enumerate(self.estimators_):
            for split in self._extract_splits(estimator[0].tree_, iteration=iter + 1):
                feature_impacts.append(split)
        
        df = pd.DataFrame(feature_impacts, columns=['iteration', 'feature', 'threshold', 'impact', 'samples'])
        return df
    

    def _extract_splits(self, tree, node_id=0, iteration=0, path=[]):
        # Check for leaf nodes
        if tree.children_left[node_id] == tree.children_right[node_id] == -1:
            return []

        # Check node's split details; not a leaf node
        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        left_child_node_id, right_child_node_id = tree.children_left[node_id], tree.children_right[node_id]
        
        # Calculate the impact of a feature split, defined as value of left child - value of right child
        impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()) * self.learning_rate

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
    
    def get_prediction_feature_impacts(self, X):
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
                impact = abs(tree.value[right_child_node_id].sum() - tree.value[left_child_node_id].sum()) * self.learning_rate

                feature_impacts.append({
                    'threshold': threshold,
                    'iteration': iter_num + 1,
                    'feature': feature_index,
                    'impact': impact
                })

        df_impacts = pd.DataFrame(feature_impacts)
        return df_impacts

    def get_predictions_feature_impacts(self, X):
        """
        Takes in mulitple data points and consolidates feature impacts on predictions
        """
        dfs = []
        for i in range(len(X)):
            dfs.append(self.get_prediction_feature_impacts(np.array(X.iloc[i, :]).reshape(1, -1)))
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
    feature_impacts_per_iter = gbr.get_feature_impacts_per_iteration()
