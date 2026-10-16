"""
Heart Disease Risk Analysis and Prediction Using Deep Learning
Dataset: Framingham Heart Study dataset (Kaggle).
Place the CSV in data/ and rename it to framingham.csv.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, roc_auc_score
)
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = Path("framingham.csv")
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# 1. Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "Dataset missing. Download the Framingham Heart Study CSV using the link "
        "in README.md, then save it as data/framingham.csv."
    )

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
print("Dataset shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nFirst five rows:\n", df.head())
print("\nMissing values:\n", df.isna().sum().sort_values(ascending=False))
print("\nDuplicate rows:", df.duplicated().sum())

# 2. Find target column (different copies use different capitalization)
target = next((c for c in df.columns if c.lower() in
               {"tenyearchd", "ten_year_chd", "chd", "target"}), None)
if target is None:
    raise ValueError("Target column not found. Expected TenYearCHD or Ten_Year_CHD.")

df = df.drop_duplicates().copy()
df = df.dropna(subset=[target]).copy()
y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()
df, y = df.loc[valid].copy(), y.loc[valid].astype(int)
X = df.drop(columns=[target])

# Drop identifier columns, if present.
id_cols = [c for c in X.columns if c.lower() in {"id", "patientid", "patient_id"}]
X = X.drop(columns=id_cols, errors="ignore")

# Remove completely empty and constant columns.
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=False) > 1]

print("\nRows after basic cleaning:", len(X))
print("Input fields used:", X.shape[1])
print("Target distribution:\n", y.value_counts())

# 3. EDA visualizations (minimum 5)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(x=y)
plt.title("Ten-Year CHD Outcome Distribution")
plt.xlabel("TenYearCHD (0 = No, 1 = Yes)")
plt.ylabel("Number of participants")
plt.tight_layout()
plt.savefig(OUT / "01_target_distribution.png", dpi=160)
plt.close()

numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

if numeric_cols:
    col = numeric_cols[0]
    plt.figure(figsize=(7, 5))
    sns.histplot(X[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(OUT / "02_numeric_histogram.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.boxplot(x=X[col])
    plt.title(f"Outlier Check for {col}")
    plt.tight_layout()
    plt.savefig(OUT / "03_numeric_boxplot.png", dpi=160)
    plt.close()

if len(numeric_cols) >= 2:
    plt.figure(figsize=(11, 8))
    sns.heatmap(X[numeric_cols].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(OUT / "04_correlation_heatmap.png", dpi=160)
    plt.close()

if categorical_cols:
    col = categorical_cols[0]
    vals = X[col].fillna("Unknown").astype(str)
    order = vals.value_counts().head(10).index
    plt.figure(figsize=(8, 5))
    sns.countplot(y=vals[vals.isin(order)], order=order)
    plt.title(f"Category Frequency: {col}")
    plt.tight_layout()
    plt.savefig(OUT / "05_category_frequency.png", dpi=160)
    plt.close()
elif numeric_cols:
    col = numeric_cols[0]
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=y, y=X[col])
    plt.title(f"{col} by CHD Outcome")
    plt.tight_layout()
    plt.savefig(OUT / "05_feature_by_target.png", dpi=160)
    plt.close()

# Compare a numeric feature by target for another relationship plot.
if numeric_cols:
    col = numeric_cols[min(1, len(numeric_cols)-1)]
    plot_df = pd.DataFrame({col: X[col], "TenYearCHD": y}).dropna()
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=plot_df, x="TenYearCHD", y=col)
    plt.title(f"{col} by Ten-Year CHD Outcome")
    plt.tight_layout()
    plt.savefig(OUT / "06_feature_target_relationship.png", dpi=160)
    plt.close()

# 4. Split data into training, validation, and test sets.
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp
)

num_features = X_train.select_dtypes(include=np.number).columns.tolist()
cat_features = X_train.select_dtypes(exclude=np.number).columns.tolist()

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

prep = ColumnTransformer([
    ("numeric", num_pipe, num_features),
    ("categorical", cat_pipe, cat_features)
])

X_train_p = prep.fit_transform(X_train).astype("float32")
X_val_p = prep.transform(X_val).astype("float32")
X_test_p = prep.transform(X_test).astype("float32")

# 5. Deep learning: Multilayer Perceptron (MLP)
model = models.Sequential([
    layers.Input(shape=(X_train_p.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.30),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.20),
    layers.Dense(1, activation="sigmoid")
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

stop = callbacks.EarlyStopping(
    monitor="val_loss", patience=6, restore_best_weights=True
)
history = model.fit(
    X_train_p, y_train,
    validation_data=(X_val_p, y_val),
    epochs=40, batch_size=64, callbacks=[stop], verbose=1
)

# 6. Evaluation
prob = model.predict(X_test_p, verbose=0).ravel()
pred = (prob >= 0.5).astype(int)
acc = accuracy_score(y_test, pred)
auc_score = roc_auc_score(y_test, prob)

print(f"\nTest accuracy: {acc:.4f}")
print(f"Test ROC-AUC: {auc_score:.4f}")
print("\nClassification report:\n", classification_report(y_test, pred, zero_division=0))

# Loss vs epoch
plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Training loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Binary cross-entropy loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "07_loss_vs_epoch.png", dpi=160)
plt.close()

# Accuracy vs epoch
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "08_accuracy_vs_epoch.png", dpi=160)
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No CHD", "CHD"], yticklabels=["No CHD", "CHD"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "09_confusion_matrix.png", dpi=160)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, prob)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"MLP (AUC = {auc_score:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "10_roc_curve.png", dpi=160)
plt.close()

model.save(OUT / "heart_risk_mlp.keras")
pd.DataFrame({
    "metric": ["test_accuracy", "test_roc_auc", "test_rows", "input_features"],
    "value": [acc, auc_score, len(X_test), X.shape[1]]
}).to_csv(OUT / "metrics.csv", index=False)

print(f"\nDone. Outputs saved to: {OUT.resolve()}")
print("Educational project only; this model is not a medical diagnosis tool.")
