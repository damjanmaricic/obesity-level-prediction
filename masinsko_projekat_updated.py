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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# =========================
# 1. Učitavanje podataka
# =========================
df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

# Target kolona i ulazne promenljive
y = df["NObeyesdad"]
X = df.drop("NObeyesdad", axis=1)

# Pretvaranje naziva klasa u brojeve
le = LabelEncoder()
y = le.fit_transform(y)

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

print("Broj ulaznih promenljivih:", X_train.shape[1])
print("Broj trening uzoraka:", X_train.shape[0])
print("Broj test uzoraka:", X_test.shape[0])

# Prepoznavanje tipova ulaznih kolona
numeric_features = X_train.select_dtypes(include="number").columns
categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns

print("Numeričke kolone:", list(numeric_features))
print("Kategorijske kolone:", list(categorical_features))

# Različita obrada za numeričke i kategorijske kolone
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical_features
        )
    ]
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

# Pipeline: priprema podataka → klasifikacioni model
pipelines = {}

for name, model in models.items():
    pipelines[name] = Pipeline([
        ("preprocessor", preprocessor),
        ("reduction", "passthrough"),
        ("model", model)
    ])

# Dodavanje prefiksa "model__" parametrima za Pipeline
pipeline_grids = {}

for name, grid in param_grids.items():
    if isinstance(grid, list):
        pipeline_grids[name] = [
            {f"model__{key}": value for key, value in option.items()}
            for option in grid
        ]
    else:
        pipeline_grids[name] = {
            f"model__{key}": value
            for key, value in grid.items()
        }
# GridSearch će uporediti obradu bez LDA i sa LDA
for name in pipeline_grids:
    grid = pipeline_grids[name]

    if isinstance(grid, list):
        for option in grid:
            option["reduction"] = [
                "passthrough",
                LDA(n_components=6)
            ]
    else:
        grid["reduction"] = [
            "passthrough",
            LDA(n_components=6)
        ]
# =========================
# 3. Izbor i evaluacija modela
# =========================

results = []

for name, pipeline in pipelines.items():
    print(f"\n=== GridSearchCV: {name} ===")

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=pipeline_grids[name],
        cv=5,
        scoring="accuracy",
        n_jobs=1
    )

    # Obrada, LDA i model uče se unutar cross-validation-a
    grid.fit(X_train, y_train)

    # Najbolja kombinacija prema cross-validation-u
    best_pipeline = grid.best_estimator_
    best_reduction = best_pipeline.named_steps["reduction"]

    if best_reduction == "passthrough":
        best_space = "Without LDA"
    else:
        best_space = "With LDA"

    print("Izabrani prostor:", best_space)
    print("Najbolja cross-validation tačnost:", grid.best_score_)
    print("Najbolji parametri:", grid.best_params_)

    # Evaluacija odabrane kombinacije na test skupu
    y_pred = best_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Selected space": best_space,
        "CV accuracy": grid.best_score_,
        "Test accuracy": accuracy,
        "Test precision": precision,
        "Test recall": recall,
        "Test F1-score": f1
    })

    # Konfuziona matrica na test skupu
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_
    )
    plt.xlabel("Predviđeno")
    plt.ylabel("Stvarno")
    plt.title(f"Konfuziona matrica – {name} ({best_space})")
    plt.tight_layout()
    plt.show()

# Pregled rezultata svih modela
results_df = pd.DataFrame(results)

print("\n=== Rezultati modela ===")
print(results_df.to_string(index=False))