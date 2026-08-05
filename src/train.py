from src.pipeline.training_pipeline import TrainingPipeline

if __name__ == "__main__":

    path = input("Dataset Path : ")

    pipeline = TrainingPipeline()

    pipeline.run_pipeline(path)

    print("Training Completed Successfully")