from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).parent
README = (BASE_DIR / "README.md").read_text(encoding="utf-8")


setup(
    name="enterprise-automl-platform",
    version="2.0.0",
    author="Hridhaan Singh Dhamani",
    author_email="hridhaansinghdhamani@gmail.com",
    description=(
        "Production-Ready Enterprise AutoML Platform built with "
        "Scikit-Learn, XGBoost, LightGBM, CatBoost, "
        "Optuna, MLflow, FastAPI and Streamlit."
    ),
    long_description=README,
    long_description_content_type="text/markdown",
    python_requires=">=3.11,<3.12",
    packages=find_packages(
        exclude=(
            "tests",
            "tests.*",
            "notebooks",
            "docs",
            "artifacts",
            "saved_models",
            "reports",
        )
    ),
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    keywords=[
        "automl",
        "machine-learning",
        "mlops",
        "data-science",
        "fastapi",
        "streamlit",
        "optuna",
        "mlflow",
        "xgboost",
        "lightgbm",
        "catboost",
    ],
    project_urls={
        "Source": "https://github.com/hridhaansinghdhamani/enterprise-automl-platform",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)