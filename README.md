# forensic-bind
Forensic Embedding Space for Deepfake Attribution

## Project Structure

This project is organized to separate concerns among the 4 team members.

```
forensic-bind/
├── configs/               # Configuration files (YAML, JSON, etc.)
├── data/                  # Data storage (ignored by git usually)
│   ├── processed/         # Preprocessed data ready for training
│   └── raw/               # Original immutable data dump
├── notebooks/             # Jupyter notebooks for exploration and prototyping
├── reports/               # Generated analysis, figures, and final reports
├── scripts/               # Executable scripts (e.g., run_training.sh)
├── src/                   # Source code
│   ├── data/              # [Member 1] Data loading, preprocessing, and pipelines
│   ├── detection/         # [Member 3] Deepfake detection logic and inference
│   ├── evaluation/        # [Member 4] Metrics (Accuracy, F1), reporting tools
│   ├── models/            # [Shared] Model architectures and definitions
│   ├── training/          # [Member 2] Training loops, experiments, baselines
│   └── utils/             # [Shared] Common utility functions
├── tests/                 # Unit tests
└── README.md
```

## Responsibilities

### 1. Data, Preprocessing, and Pipeline Setup
**Folder:** `src/data/`
- **Responsibilities:**
  - Implementing data loaders (`Dataset` classes).
  - Data cleaning and augmentation pipelines.
  - Scripts to convert raw data in `data/raw/` to `data/processed/`.
  - Ensuring data consistency and integrity.

### 2. Training, Experiments, and Baselines
**Folder:** `src/training/`
- **Responsibilities:**
  - Writing the training loop (trainer classes).
  - Setting up experiment tracking (e.g., TensorBoard, WandB).
  - Implementing baseline methods for comparison.
  - Hyperparameter tuning configurations (stored in `configs/`).

### 3. Actual Deepfake Detection
**Folder:** `src/detection/`
- **Responsibilities:**
  - Implementing the core detection logic/inference pipeline.
  - APIs or scripts that take an image/video and output a prediction.
  - Integrating the trained models from `src/models/` for production/demo use.

### 4. Metrics and Reporting
**Folder:** `src/evaluation/` & `reports/`
- **Responsibilities:**
  - Implementing evaluation metrics (Accuracy, F1, AUC, etc.).
  - Generating confusion matrices and ROC curves.
  - Writing scripts to generate reports in `reports/`.
  - Analyzing model performance and errors.

## Usage

To run the project, ensure dependencies are installed and use the scripts in `scripts/` or `src/`.
