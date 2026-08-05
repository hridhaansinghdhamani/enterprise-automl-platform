import pandas as pd

from src.pipeline.prediction_pipeline import PredictionPipeline

if __name__ == "__main__":

    path = input("CSV Path : ")

    df = pd.read_csv(path)

    pipeline = PredictionPipeline()

    prediction = pipeline.predict(df)

    print(prediction)