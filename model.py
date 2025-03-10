import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# === 1. Načtení dat ===
df = pd.read_csv("keyboard_final_data.csv")

# === 2. Příprava dat ===
X = df.drop(columns=["pressing"])  # Vstupní vlastnosti
y = df["pressing"]  # Cílová proměnná (label)

# Rozdělení na trénovací (80%) a testovací (20%) sadu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 3. Trénování modelu ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === 4. Testování modelu ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "keyboard_model.pkl")
print("Model byl uložen jako keyboard_model.pkl")

print(f"Přesnost modelu: {accuracy * 100:.2f}%")
