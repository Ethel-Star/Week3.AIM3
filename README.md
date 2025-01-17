
# Insurance Planning and Data Analysis

Welcome to the Insurance Planning and Data Analysis project! This project aims to optimize marketing strategies and identify low-risk targets for premium adjustments in the insurance sector. We are using Data Version Control (DVC) to manage and track changes in our datasets and facilitate efficient collaboration.

## Data Version Control (DVC)

Data Version Control (DVC) is a free, open-source tool designed to manage large datasets, automate machine learning (ML) pipelines, and streamline experiment management. By integrating DVC with Git, we ensure reproducibility, efficient data management, and secure collaboration.

### Key Features of DVC:
- **Codification**: Defines all aspects of the ML project (data versions, ML pipelines, and experiments) in human-readable metafiles, aligning with best practices and existing engineering tools.
- **Versioning**: Utilizes Git to version and share the entire ML project, including source code, configurations, and data assets. DVC metafiles act as placeholders for actual data files.
- **Secure Collaboration**: Allows control over project access, enabling secure sharing with selected collaborators.

## Getting Started

Follow these steps to set up and use DVC in this project:

1. **Install DVC**:
   ```bash
   pip install dvc
dvc init
mkdir /path/to/your/local/storage
dvc remote add -d localstorage /path/to/your/local/storage
dvc add raw_data.csv
dvc add cleaned_data.csv
git add raw_data.csv.dvc cleaned_data.csv.dvc .gitignore
git commit -m "Add raw and cleaned data files with DVC tracking"
dvc push
### Project Structure 

├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   └── utils.py
├── notebooks/
│   ├── __init__.py
│   └── insurance_EDA.ipynb
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md

