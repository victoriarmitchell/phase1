from setuptools import setup, find_packages

setup(
    name="fraud-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "google-cloud-storage",
        "google-cloud-aiplatform",
        "joblib"
    ]
)