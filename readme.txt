# Domain-Adaptive Machine Learning for Accurate Cure Degree Prediction in Additive Manufacturing with Imbalanced Data


This repository accompanies our research paper on applying domain-specific constraints and adaptive weighting to improve classification performance on imbalanced, physics-governed datasets.

## 🧪 Dataset

- `data/categorized_cure_data.csv`: Contains labeled process parameter data with "Low", "Moderate", and "High" cure categories.

## 🧠 Model

- `src/model.py`: Implementation of the `PhysicsConstrainedADWB_RF` classifier. A custom random forest classifier that oversamples based on physically valid process constraints and optimal parameter regions.
- `src/utils.py`: Helper functions including a safe SMOTE wrapper.
- `src/train.py`: Full training pipeline, comparing baseline Random Forest, SMOTE-RF, and our proposed model.

## 📊 Results

- `results/performance.txt`: Classification reports and confusion matrices for each model.

## 🚀 Getting Started

Install dependencies:
```bash
pip install -r requirements.txt

To train and evaluate all models:

python src/train.py

## 📚 Citation
If you use this work, please cite our upcoming research paper (DOI or citation to be added after publication).

## 📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.
