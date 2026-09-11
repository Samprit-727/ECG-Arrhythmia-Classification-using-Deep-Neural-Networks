# ECG Heartbeat Classification

This project classifies heartbeat signals from the MIT-BIH Arrhythmia dataset into five categories. It is deliberately kept at an intermediate level: three classical machine-learning models and two deep-learning models.

## Models

* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Artificial Neural Network (ANN)
* 1D Convolutional Neural Network (CNN)

There is no voting classifier, LSTM, Transformer, or model ensemble.

## Dataset layout

Each CSV row has 187 ECG signal samples and one final class label. The project expects:

* `data/raw/heartbeat/mitbih\_train.csv` — downloaded from Kaggle
* `dataset hb/mitbih\_test.csv` — supplied test file

Labels: `0` Normal, `1` Supraventricular, `2` Ventricular, `3` Fusion, `4` Unclassifiable.

## Method

1. Split the original training data into training and validation sets.
2. Balance only the training subset (default: 5,000 samples per class).
3. Standardize signal values using `StandardScaler`.
4. Train each model.
5. Evaluate once on the original, untouched test data.

Keeping the test set unbalanced makes the final results more realistic.

## Measured results

The following results were produced on the untouched test set using `--samples-per-class 2000 --epochs 10`. They are an initial benchmark, not a final clinical-performance claim.

|Model|Accuracy|Weighted Precision|Weighted Recall|Weighted F1-score|
|-|-:|-:|-:|-:|
|Random Forest|93.50%|95.48%|93.50%|94.19%|
|CNN|89.96%|94.66%|89.96%|91.63%|
|SVM|87.40%|93.85%|87.40%|89.76%|
|ANN|82.46%|93.37%|82.46%|86.33%|
|Decision Tree|81.34%|90.98%|81.34%|84.48%|

For this classification task, RMSE is not used. Accuracy, precision, recall, F1-score, and confusion matrices are the appropriate metrics.

## Run

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python train\_ecg\_models.py
```

If your Windows installation uses the Python launcher, replace `python` with `py`.

Useful smaller first run:

```powershell
python train\_ecg\_models.py --samples-per-class 2000 --epochs 10
```

## Outputs

The `outputs/` folder will contain:

* `model\_comparison.csv` — accuracy, weighted precision, recall, and F1 comparison
* one classification report and normalized confusion matrix per model
* `ann\_model.keras` and `cnn\_model.keras`

## Streamlit demo

After training, start the interactive app with:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Upload a CSV containing 187 ECG signal values per row. The supplied test CSV can be used for a demonstration. The app plots a selected ECG signal, predicts its class, shows probabilities, and displays the saved model comparison.

After adding this feature, rerun the training command once. It saves the scaler and classical ML models needed by the app.
