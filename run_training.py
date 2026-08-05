from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":

    TrainingPipeline().run_pipeline(
        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )