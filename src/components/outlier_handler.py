import numpy as np


class OutlierHandler:

    def iqr(self, dataframe, columns):

        df = dataframe.copy()

        for col in columns:

            q1 = df[col].quantile(0.25)

            q3 = df[col].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr

            upper = q3 + 1.5 * iqr

            df[col] = np.where(
                df[col] < lower,
                lower,
                np.where(df[col] > upper, upper, df[col])
            )

        return df