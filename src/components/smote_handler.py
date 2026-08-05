from collections import Counter

from imblearn.over_sampling import SMOTE

from src.constants.application import IMBALANCE_THRESHOLD


class SMOTEHandler:

    def apply(self, X, y):

        class_count = Counter(y)

        majority = max(class_count.values())
        minority = min(class_count.values())

        imbalance = minority / majority

        if imbalance < IMBALANCE_THRESHOLD:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)

        return X, y