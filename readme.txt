# Physics-Constrained Adaptive Domain Weighting with Random Forest

This repository accompanies our research paper on applying domain-specific constraints and adaptive weighting to improve classification performance on imbalanced, physics-governed datasets.

## 🧪 Dataset

- `data/categorized_cure_data.csv`: Contains labeled process parameter data with "Low", "Moderate", and "High" cure categories.

## 🧠 Model

- `src/model.py`: Implementation of the `PhysicsConstrainedADWB_RF` classifier.
- `src/utils.py`: Helper functions including a safe SMOTE wrapper.
- `src/train.py`: Full training pipeline, comparing baseline Random Forest, SMOTE-RF, and our proposed model.

## 📊 Results

- `results/performance.txt`: Classification reports and confusion matrices for each model.

## 🚀 Getting Started

Install dependencies:
```bash
pip install -r requirements.txt
