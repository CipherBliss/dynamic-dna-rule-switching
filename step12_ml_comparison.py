import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------- CONFIGURATION ----------------
# We will compare how easy it is to detect each method
methods_to_test = ["Entropy", "Chaotic", "Hybrid"]
classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Load Data
df = pd.read_csv("stego_features.csv")

results = {
    "Model": [],
    "Stego Method": [],
    "Accuracy": []
}

print("📊 RUNNING ML STEGANALYSIS...\n")

# ---------------- TRAINING LOOP ----------------
for stego_method in methods_to_test:
    print(f"--- Analyzing Security of: {stego_method} ---")
    
    # Filter data: Get only Cover (Label 0) and specific Stego (Label 1)
    subset = df[df["Method"] == stego_method]
    
    X = subset[["Mean", "Variance", "Skewness", "Kurtosis", "Entropy"]]
    y = subset["Label"]
    
    # Split 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    for clf_name, clf in classifiers.items():
        # Train
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        
        # Calculate Accuracy
        acc = accuracy_score(y_test, y_pred)
        results["Model"].append(clf_name)
        results["Stego Method"].append(stego_method)
        results["Accuracy"].append(acc)
        
        print(f"   {clf_name}: {acc*100:.2f}% Detection Accuracy")

# ---------------- PLOTTING RESULTS ----------------
results_df = pd.DataFrame(results)

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x="Stego Method", y="Accuracy", hue="Model", palette="viridis")

plt.axhline(y=0.5, color='r', linestyle='--', label="Ideal Security (50%)")
plt.title("Steganalysis Detection Accuracy (Lower is Better for Steganography)")
plt.ylabel("Detection Accuracy (0.5 = Random/Secure, 1.0 = Detected)")
plt.ylim(0, 1.05)
plt.legend(title="ML Classifier")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✅ ML Comparison Graph Generated.")
print("\nINTERPRETATION:")
print("1. Low Accuracy (~0.50): The method is SECURE. The AI is guessing.")
print("2. High Accuracy (>0.80): The method is WEAK. The AI can detect the changes.")
print("3. Compare if Hybrid has lower detection accuracy than Entropy/Chaotic.")