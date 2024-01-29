from sklearn.ensemble import GradientBoostingRegressor
class RegressorTreeProcessor:
    def __init__(self, gb_regressor):
        if not isinstance(gb_regressor, GradientBoostingRegressor):
            raise ValueError("Input must be a GradientBoostingRegressor instance.")
        self.gb_regressor = gb_regressor
        self.trees = gb_regressor.estimators_

    def get_tree(self, tree_index):
        if tree_index < 0 or tree_index >= len(self.trees):
            raise ValueError(f"Tree index must be between 0 and {len(self.trees) - 1}.")
        return self.trees[tree_index][0]

    def predict_with_tree(self, tree_index, X):
        tree = self.get_tree(tree_index)
        return tree.predict(X)
    
    def track_tree_prediction(self, tree_index, X):
        tree = self.get_tree(tree_index)
        node_values = []
        node = tree.tree_
        current_node = 0
        while node.children_left[current_node] != node.children_right[current_node]:
            feature_index = node.feature[current_node]
            threshold = node.threshold[current_node]
            if X[0, feature_index] <= threshold:
                current_node = node.children_left[current_node]
            else:
                current_node = node.children_right[current_node]
            node_values.append((feature_index, threshold))
        return node_values