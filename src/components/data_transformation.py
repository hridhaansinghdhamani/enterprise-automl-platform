from src.components.preprocessor import Preprocessor


class DataTransformation:

    def __init__(self):

        self.preprocessor = Preprocessor()

    def transform(

        self,

        X_train,

        X_test,

        numerical_columns,

        categorical_columns,

    ):

        transformer = self.preprocessor.build(

            numerical_columns,

            categorical_columns,

        )

        X_train = transformer.fit_transform(X_train)

        X_test = transformer.transform(X_test)

        return (

            X_train,

            X_test,

            transformer,

        )