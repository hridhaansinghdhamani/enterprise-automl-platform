from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureSelection:

    def select(self, X, y, k=20):

        selector = SelectKBest(
            score_func=mutual_info_classif,
            k=min(k, X.shape[1])
        )

        X = selector.fit_transform(X, y)

        return X, selector