"""
Enterprise AutoML Platform
Author: Hridhaan Singh Dhamani
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.logger.logger import get_logger
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline

logger = get_logger(__name__)

st.set_page_config(
    page_title="Enterprise AutoML Platform",
    page_icon="🤖",
    layout="wide",
)

st.title("🚀 Enterprise AutoML Platform")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Train Model",
        "Batch Prediction",
    ],
)

# ======================================================
# TRAIN MODEL
# ======================================================

if menu == "Train Model":

    file = st.file_uploader(
        "Upload Training Dataset",
        type=["csv"],
    )

    if file is not None:

        try:

            df = pd.read_csv(file)

            if df.empty:
                st.error("Uploaded dataset is empty.")
                st.stop()

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(),
                use_container_width=True,
            )

            st.info(
                f"Rows: {df.shape[0]} | Columns: {df.shape[1]}"
            )

            if st.button(
                "🚀 Start Training",
                use_container_width=True,
            ):

                Path("data").mkdir(
                    exist_ok=True,
                )

                upload_path = "data/upload.csv"

                df.to_csv(
                    upload_path,
                    index=False,
                )

                with st.spinner(
                    "Training model..."
                ):

                    result = (
                        TrainingPipeline()
                        .run_pipeline(upload_path)
                    )

                st.success(
                    "Training completed successfully."
                )

                st.subheader(
                    "🏆 Model Leaderboard"
                )

                st.dataframe(
                    result["leaderboard"],
                    use_container_width=True,
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Best Model",
                        result["best_model"],
                    )

                with col2:

                    st.metric(
                        "Best CV Score",
                        round(
                            result["best_score"],
                            4,
                        ),
                    )

                logger.info(
                    "Training completed successfully."
                )

        except Exception as e:

            logger.exception(
                "Training failed."
            )

            st.error(
                f"Training failed.\n\n{str(e)}"
            )

# ======================================================
# BATCH PREDICTION
# ======================================================

elif menu == "Batch Prediction":

    file = st.file_uploader(
        "Upload Prediction Dataset",
        type=["csv"],
        key="prediction",
    )

    if file is not None:

        try:

            df = pd.read_csv(file)

            if df.empty:
                st.error("Uploaded dataset is empty.")
                st.stop()

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                df.head(),
                use_container_width=True,
            )

            with st.spinner(
                "Generating predictions..."
            ):

                result = (
                    PredictionPipeline()
                    .predict(df)
                )

            st.success(
                "Prediction completed successfully."
            )

            st.subheader(
                "Prediction Result"
            )

            st.dataframe(
                result,
                use_container_width=True,
            )

            st.download_button(
                label="📥 Download Predictions",
                data=result.to_csv(
                    index=False,
                ),
                file_name="prediction.csv",
                mime="text/csv",
                use_container_width=True,
            )

            logger.info(
                "Batch prediction completed."
            )

        except Exception as e:

            logger.exception(
                "Prediction failed."
            )

            st.error(
                f"Prediction failed.\n\n{str(e)}"
            )