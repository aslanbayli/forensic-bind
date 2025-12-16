# forensic-bind
Forensic Embedding Space for Deepfake Attribution

## Project Structure

This project uses a simplified notebook-based approach, organized to separate concerns.

```
forensic-bind/
├── data/                  # Data storage 
│   ├── processed/         # Preprocessed data ready for analysis
│   └── raw/               # Original immutable data dump
├── reports/               # Generated analysis, figures, and final reports
├── src/                   # Jupyter notebooks organized by responsibility
│   ├── data/              # Data loading and preprocessing notebooks
│   ├── evaluation/        # Metrics and analysis notebooks
│   ├── models/            # Trained models
│   ├── training/          # Training and experiment notebooks
│   └── utils/             # Utility functions and helpers
└── README.md
```

## Responsibilities

### 1. Data and Preprocessing
**Folder:** `src/data/`
- **Responsibilities:**
  - Data loading and exploration notebooks
  - Data cleaning and augmentation
  - Converting raw data in `data/raw/` to `data/processed/`
  - Ensuring data consistency and integrity

### 2. Training and Experiments
**Folder:** `src/training/`
- **Responsibilities:**
  - Training notebooks with experiment tracking
  - Implementing baseline methods for comparison
  - Hyperparameter tuning and ablation studies
  - Training models and saving them to `src/models/`

### 3. Model Architectures
**Folder:** `src/models/`
- **Responsibilities:**
  - Trained models

### 4. Metrics and Evaluation
**Folder:** `src/evaluation/`
- **Responsibilities:**
  - Evaluation metrics (Accuracy, F1, etc.)
  - Confusion matrices and ROC curves
  - Model performance analysis and reporting
  - Generating reports in `reports/`
  - Analyzing model performance and errors

## Usage

1. Set up the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Launch Jupyter and open notebooks from the `src/` directory:
   ```bash
   jupyter notebook
   ```
