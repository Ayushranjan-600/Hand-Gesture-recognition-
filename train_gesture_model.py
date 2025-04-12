import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix
)
from sklearn.utils.multiclass import unique_labels


def load_dataset(filepath="gesture_data.csv"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Dataset file '{filepath}' not found!")
    df = pd.read_csv(filepath)
    if "label" not in df.columns:
        raise ValueError("❌ 'label' column missing in dataset.")
    return df


def preprocess_data(df):
    X = df.drop("label", axis=1).values
    y = df["label"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_encoded, label_encoder, scaler


def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test, label_encoder):
    y_pred = clf.predict(X_test)
    used_labels = unique_labels(y_test, y_pred)
    target_names = label_encoder.inverse_transform(used_labels)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, labels=used_labels, target_names=target_names))
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def save_model(clf, scaler, label_encoder, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(model_dir, "gesture_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join(model_dir, "label_encoder.pkl"))
    print("\n✅ Model, scaler, and label encoder saved in /models")


def show_feature_importance(clf, top_n=10):
    importances = clf.feature_importances_
    top_indices = np.argsort(importances)[-top_n:][::-1]

    print(f"\n🔥 Top {top_n} Important Features (Landmarks):")
    for idx in top_indices:
        print(f"Feature {idx}: Importance = {importances[idx]:.4f}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🚀 Training Gesture Recognition Model...")

    df = load_dataset("gesture_data.csv")
    X_scaled, y_encoded, label_encoder, scaler = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = train_model(X_train, y_train)

    evaluate_model(clf, X_test, y_test, label_encoder)
    show_feature_importance(clf)
    save_model(clf, scaler, label_encoder)
