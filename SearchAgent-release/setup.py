from setuptools import setup, find_packages

setup(
    name="searchagent",
    version="0.1.0",
    description="Behavior-Constrained LLM Search User Simulation",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",  # calibration network only; LLM is API (DeepSeek)
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
)
