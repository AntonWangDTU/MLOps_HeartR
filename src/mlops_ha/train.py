# from mlops_ha.model import Model
from mlops_ha.data import MyDataset
from pathlib import Path


import typer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler


def train(
    data_path: Path,
    output_path: Path,
    artifacts_path: Path,  # new: where to store models and mlflow db
) -> None:
    dataset = MyDataset(data_path, output_path)
    X = dataset.features
    y = dataset.targets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    )
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    # Save model
    model_path = artifacts_path / "model.pkl"
    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    typer.run(train)
