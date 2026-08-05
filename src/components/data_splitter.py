from sklearn.model_selection import train_test_split


class DataSplitter:

    def split(
        self,
        X,
        y,
        test_size,
        random_state,
        classification=True,
    ):

        if classification:

            return train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )

        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )