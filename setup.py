from setuptools import find_packages, setup

setup(
    name="network-expert-system",
    version="0.1.0",
    description="Hybrid rule-based and ML expert system for network troubleshooting",
    packages=find_packages(include=["src*", "app*"]),
    install_requires=[
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        "joblib>=1.3.0",
        "streamlit>=1.35.0",
    ],
    python_requires=">=3.10",
    entry_points={"console_scripts": ["network-expert=app.main:main"]},
)
