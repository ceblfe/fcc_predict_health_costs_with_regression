# Linear Regression Health Costs Calculator
Linear Regression Health Costs Calculator. This project is the fourth project to get the Machine Learning with Python Certification from freeCodeCamp.

## Overview
This project implements a regression model using TensorFlow to predict healthcare costs based on a dataset containing information about individuals, including their age, sex, BMI, number of children, smoking status, region, and healthcare expenses. The goal is to train a neural network to predict the `expenses` column with a Mean Absolute Error (MAE) of less than 3500 on the test set, as specified in the challenge.

The implementation is provided in the Jupyter notebook `fcc_predict_health_costs_with_regression_CBF.ipynb`.

## Dataset
The dataset (`insurance.csv`) is sourced from [FreeCodeCamp](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv) and contains the following features:
- `age`: Age of the individual (numeric).
- `sex`: Gender (categorical: `female`, `male`).
- `bmi`: Body Mass Index (numeric).
- `children`: Number of children (numeric).
- `smoker`: Smoking status (categorical: `yes`, `no`).
- `region`: Region of residence (categorical: `southwest`, `southeast`, `northwest`, `northeast`).
- `expenses`: Healthcare costs (numeric, target variable).

The dataset is split into 80% training and 20% testing sets.

## Requirements
To run the notebook, ensure you have the following dependencies installed:
- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- tensorflow_docs (installed via `!pip install -q git+https://github.com/tensorflow/docs`)

You can install the required packages in a Jupyter notebook environment (e.g., Google Colab) or locally with:
```bash
pip install tensorflow pandas numpy matplotlib
pip install git+https://github.com/tensorflow/docs
```

## How to Run
1. **Open the Notebook**: Load `fcc_predict_health_costs_with_regression_CBF.ipynb` in a Jupyter environment (e.g., Google Colab or Jupyter Notebook).
2. **Execute Cells in Order**:
   - The first cell installs and imports necessary libraries.
   - The second cell downloads the `insurance.csv` dataset.
   - The third cell preprocesses the data by converting categorical features (`sex`, `smoker`, `region`) to numerical values and splitting the dataset into training (80%) and testing (20%) sets.
   - The fourth cell creates a normalization layer for the features using `tf.keras.layers.Normalization`.
   - The fifth cell defines and builds a neural network model with a normalization layer, two dense layers (128 and 64 neurons with ReLU activation), and an output layer.
   - The sixth cell trains the model for up to 500 epochs with a validation split of 0.2.
   - The final cell evaluates the model on the test set and plots predictions against true values.
3. **Check Results**: The final cell will print the MAE and indicate whether the challenge is passed (MAE < 3500). A scatter plot of predicted vs. true expenses is also generated.

## Model Details
- **Preprocessing**:
  - Categorical features are mapped to integers: `sex` (`female=0`, `male=1`), `smoker` (`no=0`, `yes=1`), `region` (`southwest=0`, `southeast=1`, `northwest=2`, `northeast=3`).
  - Features are normalized using `tf.keras.layers.Normalization`, adapted to the training dataset.
  - The `expenses` column is used as the target variable (`train_labels`, `test_labels`).
- **Model Architecture**:
  - Input: Normalization layer to scale features.
  - Hidden Layers: Two dense layers with 128 and 64 neurons, respectively, using ReLU activation.
  - Output Layer: Single neuron for predicting `expenses`.
- **Training**:
  - Optimizer: Adam with a learning rate of 0.001.
  - Loss: Mean Squared Error (MSE).
  - Metrics: Mean Absolute Error (MAE) and MSE.
  - Epochs: Up to 500 with a validation split of 0.2 and progress displayed using `tensorflow_docs.modeling.EpochDots`.
- **Evaluation**: The model is evaluated on the test set, aiming for an MAE < 3500.

## Expected Output
After running the notebook, the final cell will output:
- The test set MAE, formatted as `Testing set Mean Abs Error: {value} expenses`.
- A message indicating whether the challenge is passed: `You passed the challenge. Great job!` (if MAE < 3500) or `The Mean Abs Error must be less than 3500. Keep trying.`
- A scatter plot comparing true expenses (`test_labels`) to predicted expenses (`test_predictions`).

## Troubleshooting
- **High MAE**: If the MAE exceeds 3500, try:
  - Increasing epochs to 1000 or adding early stopping with `tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=50, restore_best_weights=True)`.
  - Reducing the learning rate to 0.0005.
  - Simplifying the model (e.g., 64 → 32 neurons).
- **Data Issues**: Verify the dataset loads correctly (`dataset.head()`) and categorical mappings are accurate.
- **Environment**: Ensure TensorFlow 2.x is used and all dependencies are installed.

## Notes
- The `smoker` feature has a significant impact on expenses, so the model should learn this relationship effectively.
- The normalization layer ensures consistent scaling of features, improving model performance.
- This implementation is designed to run in Google Colab with GPU acceleration for faster training.