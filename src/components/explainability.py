import logging
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


logger = logging.getLogger(__name__)


class SHAPVisualizer:
    """
    SHAP Explainability Module

    Features
    --------
    - Automatic explainer selection
    - Supports classification and regression models
    - SHAP summary plot
    - SHAP feature importance plot
    - Feature importance CSV export
    - Automatic sampling for large datasets
    """

    def __init__(
        self,
        output_dir: str = "reports",
        max_samples: int = 1000,
    ):
        self.output_dir = output_dir
        self.max_samples = max_samples

        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_data(self, X):
        """
        Convert input to DataFrame and sample if dataset is large.
        """

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if len(X) > self.max_samples:
            logger.info(
                "Sampling %d rows from %d rows for SHAP.",
                self.max_samples,
                len(X),
            )

            X = X.sample(
                self.max_samples,
                random_state=42,
            )

        return X

    def _build_explainer(self, model, X):
        """
        Automatically choose the best SHAP explainer.
        """

        try:
            logger.info("Using TreeExplainer.")
            return shap.TreeExplainer(model)
        except Exception:
            pass

        try:
            logger.info("Using LinearExplainer.")
            masker = shap.maskers.Independent(X)
            return shap.LinearExplainer(model, masker)
        except Exception:
            pass

        logger.info("Using Generic Explainer.")
        return shap.Explainer(model, X)

    def generate(self, model, X):
        """
        Generate SHAP explainability reports.

        Returns
        -------
        dict | None
        """

        try:
            logger.info("Generating SHAP explanations...")

            X = self._prepare_data(X)

            explainer = self._build_explainer(model, X)

            shap_values = explainer(X)

            values = shap_values.values

            # Handle binary classification output
            if values.ndim == 3:
                values = values[:, :, 1]

            # -------------------------
            # SHAP Summary Plot
            # -------------------------
            summary_path = os.path.join(
                self.output_dir,
                "shap_summary.png",
            )

            plt.figure(figsize=(12, 8))

            shap.summary_plot(
                values,
                X,
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                summary_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            # -------------------------
            # Feature Importance Plot
            # -------------------------
            bar_path = os.path.join(
                self.output_dir,
                "feature_importance.png",
            )

            plt.figure(figsize=(10, 7))

            shap.summary_plot(
                values,
                X,
                plot_type="bar",
                show=False,
            )

            plt.tight_layout()

            plt.savefig(
                bar_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.close()

            # -------------------------
            # CSV Export
            # -------------------------
            importance = (
                pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance": np.abs(values).mean(axis=0),
                    }
                )
                .sort_values(
                    by="importance",
                    ascending=False,
                )
                .reset_index(drop=True)
            )

            csv_path = os.path.join(
                self.output_dir,
                "feature_importance.csv",
            )

            importance.to_csv(
                csv_path,
                index=False,
            )

            logger.info("SHAP reports generated successfully.")

            return {
                "summary_plot": summary_path,
                "bar_plot": bar_path,
                "importance_csv": csv_path,
                "feature_importance": importance,
            }

        except Exception as e:
            logger.exception(
                "Failed to generate SHAP explanations: %s",
                str(e),
            )
            return None