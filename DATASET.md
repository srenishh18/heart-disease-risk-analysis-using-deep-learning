# Dataset — Framingham Heart Study

**Source:** [Kaggle](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)

4,240 patients · 15 input features · Target: `TenYearCHD` (10-year CHD risk)

## Features

| Feature | Type | Description |
|---------|------|-------------|
| `male` | Binary | Sex (1 = Male, 0 = Female) |
| `age` | Integer | Age in years |
| `education` | Categorical | Education level (1–4) |
| `currentSmoker` | Binary | Current smoker |
| `cigsPerDay` | Continuous | Cigarettes per day |
| `BPMeds` | Binary | On blood pressure medication |
| `prevalentStroke` | Binary | History of stroke |
| `prevalentHyp` | Binary | Hypertension history |
| `diabetes` | Binary | Diabetes diagnosis |
| `totChol` | Continuous | Total cholesterol (mg/dL) |
| `sysBP` | Continuous | Systolic blood pressure (mmHg) |
| `diaBP` | Continuous | Diastolic blood pressure (mmHg) |
| `BMI` | Continuous | Body Mass Index (kg/m²) |
| `heartRate` | Continuous | Heart rate (beats/min) |
| `glucose` | Continuous | Blood glucose (mg/dL) |
| **`TenYearCHD`** | **Binary** | **Target — 10-year CHD risk** |

## Download

1. Go to the [Kaggle page](https://www.kaggle.com/datasets/dileep070/heart-disease-prediction-using-logistic-regression)
2. Download and extract the ZIP
3. Place `framingham.csv` in the project root

> The CSV is excluded from this repo via `.gitignore`.
