"""Setup script for the USA Economic Forecasting platform."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text(encoding="utf-8")
install_requires = [
    line.strip()
    for line in requirements.splitlines()
    if line.strip() and not line.startswith("#")
]

# Development dependencies
dev_requires = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.7.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "isort>=5.12.0",
    "ipython>=8.14.0",
    "jupyter>=1.0.0",
]

setup(
    name="usa-econ",
    version="0.1.0",
    author="Economic Forecasting Team",
    author_email="your.email@example.com",
    description="PhD-level Economic Intelligence Platform for advanced forecasting and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/economic_forecasting_usa",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/economic_forecasting_usa/issues",
        "Documentation": "https://github.com/yourusername/economic_forecasting_usa#readme",
        "Source Code": "https://github.com/yourusername/economic_forecasting_usa",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "all": dev_requires,
    },
    entry_points={
        "console_scripts": [
            "usa-econ-fetch=scripts.fetch:main",
            "usa-econ-forecast=scripts.forecast:main",
            "usa-econ-analyze=scripts.analyze:main",
            "usa-econ-professional=scripts.professional_cli:main",
            "usa-econ-phd=scripts.phd_research_cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "economics",
        "forecasting",
        "econometrics",
        "time-series",
        "machine-learning",
        "financial-analysis",
        "risk-modeling",
        "economic-indicators",
        "federal-reserve",
        "fred-api",
        "bls",
        "census",
    ],
)
