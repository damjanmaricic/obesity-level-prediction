import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Učitavanje podataka
# =========================
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Target kolona
y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

# One-hot encoding za X
X = pd.get_dummies(X, drop_first=True)

# Label encoding za y
le = LabelEncoder()
y = le.fit_transform(y)

# Standardizacija
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podela na trening i test skup (15%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)

print("Originalna dimenzionalnost:", X_train.shape[1])

# =========================
# 2. GridSearch parametri
# =========================
param_grids = {
    "LogReg": [
        {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10]},
        {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10]}
    ],
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    "Decision Tree": {
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 5]
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
}

models = {
    "LogReg": LogisticRegression(max_iter=10000, tol=1e-3, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

# =========================
# 3. GridSearch u originalnom prostoru
# =========================
results_orig = []

for name, model in models.items():
    print(f"\n=== GridSearchCV za {name} (originalni prostor) ===")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # Konfuziona matrica
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predviđeno")
    plt.ylabel("Stvarno")
    plt.title(f"Konfuziona matrica - {name} (Original)")
    plt.tight_layout()
    plt.show()

    results_orig.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-score": f1_score(y_test, y_pred, average="weighted"),
        "Space": "Original"
    })

results_orig_df = pd.DataFrame(results_orig)
print("\n=== Rezultati u originalnom prostoru ===")
print(results_orig_df)

# =========================
# 4. LDA transformacija
# =========================
lda = LDA(n_components=len(le.classes_) - 1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("Dimenzionalnost nakon LDA:", X_train_lda.shape[1])
dim_reduction = 100 * (X_train.shape[1] - X_train_lda.shape[1]) / X_train.shape[1]
print(f"Dimenzionalnost smanjena za {dim_reduction:.2f}%")

# =========================
# 5. GridSearch u LDA prostoru
# =========================
results_lda = []

for name, model in models.items():
    print(f"\n=== GridSearchCV za {name} (LDA prostor) ===")
    grid = GridSearchCV(model, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_lda, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_lda)

    # Konfuziona matrica
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Predviđeno")
    plt.ylabel("Stvarno")
    plt.title(f"Konfuziona matrica - {name} (LDA)")
    plt.tight_layout()
    plt.show()

    results_lda.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-score": f1_score(y_test, y_pred, average="weighted"),
        "Space": "LDA"
    })

results_lda_df = pd.DataFrame(results_lda)
print("\n=== Rezultati u LDA prostoru ===")
print(results_lda_df)

# =========================
# 6. Kombinovanje i poređenje
# =========================
df_compare = pd.concat([results_orig_df, results_lda_df], ignore_index=True)
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_compare, x="Model", y=metric, hue="Space")
    plt.ylim(0, 1)
    plt.title(f"Poređenje modela: {metric} (Original vs LDA)")
    plt.tight_layout()
    plt.show()

# =========================
# 7. Konačna evaluacija: izbor najboljeg prostora po modelu
# =========================
final_results = []
for name in models.keys():
    df_model = df_compare[df_compare["Model"] == name]
    best_space = df_model.loc[df_model["Accuracy"].idxmax(), "Space"]

    if best_space == "Original":
        best_X_train, best_X_test = X_train, X_test
    else:
        best_X_train, best_X_test = X_train_lda, X_test_lda

    # Treniramo model sa najboljim hiperparametrima na celom trening skupu
    grid = GridSearchCV(models[name], param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(best_X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(best_X_test)

    final_results.append({
        "Model": name,
        "Best Space": best_space,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-score": f1_score(y_test, y_pred, average="weighted")
    })

final_df = pd.DataFrame(final_results)
print("\n=== Konačni rezultati ===")
print(final_df)

# Vizualizacija finalnih rezultata
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=final_df, x="Model", y=metric)
    plt.ylim(0, 1)
    plt.title(f"Konačni rezultati: {metric}")
    plt.tight_layout()
    plt.show()
