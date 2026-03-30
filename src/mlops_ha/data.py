from pathlib import Path
import pandas as pd
import numpy as np
import typer

app = typer.Typer()


class MyDataset:
    """Simple dataset class for sklearn usage."""

    def __init__(self, data_path: Path, output_path: Path) -> None:
        self.data_path = data_path
        self.output_path = output_path
        self.df = None
        self.features = None
        self.targets = None

        if self.output_path.exists():
            df = pd.read_csv(self.output_path)
            self._load_from_df(df)
        else:
            print(f"{output_path} not found. Call preprocess() first.")

    def _load_from_df(self, df: pd.DataFrame):
        self.df = df
        self.features = df.drop(columns="num").values
        self.targets = df["num"].values

    def preprocess(self) -> None:
        """Preprocess raw data and save it."""
        df = pd.read_csv(self.data_path)

        columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "num",
        ]
        df.columns = columns

        # Select only features you care about
        df = df[["age", "sex", "cp", "trestbps", "chol", "thalach", "exang", "num"]]

        # Clean data
        df.replace("?", np.nan, inplace=True)
        df.dropna(inplace=True)
        df = df.apply(pd.to_numeric)

        # Binary target
        df["num"] = (df["num"] > 0).astype(int)

        # Save + load
        df.to_csv(self.output_path, index=False)
        self._load_from_df(df)


@app.command()
def preprocess(data_path: Path, output_path: Path):
    """CLI command to preprocess data."""
    typer.echo("Preprocessing data...")
    dataset = MyDataset(data_path, output_path)
    dataset.preprocess()
    typer.echo(f"Processed data saved to {output_path}")


if __name__ == "__main__":
    app()
