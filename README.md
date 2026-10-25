# Titanic Survival Prediction

A machine-learning project exploring passenger data from the Titanic and comparing classification models to predict whether a passenger survived.

## Project Overview

This notebook walks through the main stages of a supervised learning workflow: exploring the dataset, handling missing values, preparing features, training multiple classifiers, and evaluating their predictions.

The Titanic dataset is loaded directly from Seaborn with `sns.load_dataset("titanic")`, so a separate CSV file is not required when the dataset is available through Seaborn.

## Objectives

- Explore passenger attributes and inspect data quality.
- Review missing values, duplicates, descriptive statistics, and numeric correlations.
- Prepare features for classification by handling missing values, encoding categorical columns, and scaling features.
- Train and compare several classification algorithms.
- Evaluate predictions with accuracy, confusion matrices, and classification reports.
- Use cross-validation to get an additional view of model performance.

## Dataset

The notebook uses Seaborn's Titanic dataset, which contains passenger information and a binary survival target.

| Feature | Description |
|---|---|
| `survived` | Target: 0 = did not survive, 1 = survived |
| `pclass` | Passenger ticket class |
| `sex` | Passenger sex |
| `age` | Passenger age |
| `sibsp` | Number of siblings/spouses aboard |
| `parch` | Number of parents/children aboard |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation |
| `alone` | Whether the passenger was travelling alone |

The notebook removes several redundant or high-missingness columns, fills missing ages with the mean, and removes rows with missing embarkation values.

## Workflow

1. **Load and inspect data** — preview the data, review data types and summary statistics, and check nulls and duplicates.
2. **Explore relationships** — visualize correlations between numeric features with a heatmap.
3. **Clean the data** — remove selected columns, impute missing ages, and drop rows missing `embarked`.
4. **Prepare features** — separate `survived` as the target, one-hot encode categorical features, and standardize the resulting feature matrix.
5. **Train and evaluate models** — use an 80/20 train-test split with `random_state=42`.
6. **Compare classifiers** — evaluate Logistic Regression, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, Decision Tree, and Support Vector Classifier (SVC).
7. **Cross-validation** — run 5-fold accuracy cross-validation for the SVC workflow.

## Models and Recorded Results

The notebook's saved outputs show the following test-set accuracy values:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 79.78% |
| K-Nearest Neighbors (KNN) | 80.34% |
| Gaussian Naive Bayes | 78.09% |

Decision Tree and SVC evaluation code is also included in the notebook. Results can vary with library versions or when the notebook is rerun.

## Tools and Libraries

- Python
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Jupyter Notebook / Google Colab

## Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/1kunal2006/Titanic-project.git
   cd Titanic-project
   ```

2. Install the required libraries:

   ```bash
   pip install numpy pandas seaborn matplotlib scikit-learn jupyter
   ```

3. Open `titanic.ipynb` in Jupyter Notebook or upload it to Google Colab.
4. Run the cells in order. The notebook loads the Titanic dataset using Seaborn.

## Notes

- The notebook is an educational classification project, not a production-ready prediction system.
- The preprocessing and scaling are performed before the train-test split in the current notebook. For a more rigorous evaluation, fit preprocessing steps only on the training data, ideally within a Scikit-learn pipeline.
- The cross-validation section uses the already prepared feature matrix; a pipeline would also help avoid data leakage during cross-validation.

## Author

Kunal
