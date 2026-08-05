import yaml


class SchemaGenerator:

    def generate(self, dataframe, target_column, save_path):

        schema = {
            "columns": {},
            "target_column": target_column,
            "numerical_columns": [],
            "categorical_columns": [],
        }

        for col in dataframe.columns:

            schema["columns"][col] = str(dataframe[col].dtype)

            if col == target_column:
                continue

            if dataframe[col].dtype == "object":
                schema["categorical_columns"].append(col)
            else:
                schema["numerical_columns"].append(col)

        with open(save_path, "w") as file:
            yaml.dump(schema, file)