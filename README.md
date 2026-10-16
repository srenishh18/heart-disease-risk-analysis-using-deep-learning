# Heart Disease Risk Analysis — Deep Learning

Framingham Heart Study dataset used for 10-year coronary heart disease (CHD) prediction with a TensorFlow MLP model.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Download the dataset first — see [DATASET.md](DATASET.md).

## What it does

- Loads and cleans the Framingham dataset
- Generates 6 EDA visualizations
- Trains an MLP neural network (TensorFlow/Keras)
- Evaluates with confusion matrix, ROC-AUC, and loss/accuracy curves
- Saves all outputs to `outputs/`
