"""PyTorch LSTM climate forecasting model with multi-step horizon capability."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.settings import settings
from src.models.base import ClimateModel
from src.utils.logging import logger


def set_reproducible_seed(seed: int = 42) -> None:
    """Set global seeds for reproducibility across NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LSTMModule(nn.Module):
    """PyTorch LSTM network architecture."""

    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, features)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred.squeeze(-1)


class PyTorchLSTMClimateModel(ClimateModel):
    """PyTorch LSTM Time-Series Climate Model implementation."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        self.params = params or {
            "sequence_length": 24,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "patience": 15,
            "seed": settings.seed,
        }
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.model: LSTMModule | None = None
        self._is_fitted = False
        self.training_meta: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "LSTM"

    def _create_sequences(
        self, X_data: np.ndarray, y_data: np.ndarray, seq_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build (N, L, F) 3D sequence windows strictly without boundary leakage."""
        X_seq, y_seq = [], []
        for i in range(len(X_data) - seq_len):
            X_seq.append(X_data[i : i + seq_len])
            y_seq.append(y_data[i + seq_len])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "PyTorchLSTMClimateModel":
        set_reproducible_seed(self.params.get("seed", settings.seed))
        self.feature_names = list(X_train.columns)

        X_train_clean = X_train.fillna(0.0)
        X_tr_scaled = self.scaler.fit_transform(X_train_clean)

        seq_len = int(self.params.get("sequence_length", 24))
        X_tr_seq, y_tr_seq = self._create_sequences(X_tr_scaled, y_train.values, seq_len)

        if len(X_tr_seq) == 0:
            raise ValueError(f"Training slice too short ({len(X_train)}) for sequence_length={seq_len}")

        train_ds = TensorDataset(torch.tensor(X_tr_seq), torch.tensor(y_tr_seq))
        train_loader = DataLoader(train_ds, batch_size=int(self.params.get("batch_size", 32)), shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_clean = X_val.fillna(0.0)
            X_val_scaled = self.scaler.transform(X_val_clean)
            X_v_seq, y_v_seq = self._create_sequences(X_val_scaled, y_val.values, seq_len)
            if len(X_v_seq) > 0:
                val_ds = TensorDataset(torch.tensor(X_v_seq), torch.tensor(y_v_seq))
                val_loader = DataLoader(val_ds, batch_size=int(self.params.get("batch_size", 32)), shuffle=False)

        input_dim = X_train.shape[1]
        self.model = LSTMModule(
            input_dim=input_dim,
            hidden_size=int(self.params.get("hidden_size", 64)),
            num_layers=int(self.params.get("num_layers", 2)),
            dropout=float(self.params.get("dropout", 0.2)),
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.params.get("learning_rate", 0.001)))
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience = int(self.params.get("patience", 15))
        patience_counter = 0

        self.model.train()
        epochs = int(self.params.get("epochs", 50))

        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for v_X, v_y in val_loader:
                        v_preds = self.model(v_X)
                        val_loss += float(criterion(v_preds, v_y).item()) * len(v_X)
                val_loss /= len(X_v_seq)
                self.model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info("LSTM early stopping triggered at epoch %d", epoch + 1)
                        break

        self._is_fitted = True
        self.training_meta = {
            "name": self.name,
            "params": self.params,
            "feature_names": self.feature_names,
            "trained_at": datetime.utcnow().isoformat(),
            "best_val_loss": best_val_loss if best_val_loss != float("inf") else 0.0,
        }
        logger.info("PyTorch LSTM training complete.")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted or self.model is None:
            raise RuntimeError("LSTM model must be fitted before predict()")

        seq_len = int(self.params.get("sequence_length", 24))
        X_sub = X[self.feature_names].fillna(0.0)
        X_scaled = self.scaler.transform(X_sub)

        # Build sequence windows
        if len(X_scaled) < seq_len:
            # Pad beginning if short sample
            pad_len = seq_len - len(X_scaled)
            pad = np.tile(X_scaled[0], (pad_len, 1))
            X_scaled = np.vstack([pad, X_scaled])

        X_seq = []
        for i in range(len(X_scaled) - seq_len + 1):
            X_seq.append(X_scaled[i : i + seq_len])

        X_tensor = torch.tensor(np.array(X_seq, dtype=np.float32))

        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).numpy()

        if len(preds) < len(X):
            # Align padding length
            pad_preds = np.full(len(X) - len(preds), preds[0] if len(preds) > 0 else 0.0)
            preds = np.concatenate([pad_preds, preds])

        return np.asarray(preds, dtype=float)

    def predict_multi_step(self, X_history: pd.DataFrame, steps: int = 12) -> np.ndarray:
        """Recursive multi-step forecast over horizon h."""
        preds = []
        current_df = X_history.copy()
        for _ in range(steps):
            next_pred = self.predict(current_df)[-1]
            preds.append(float(next_pred))

            # Shift features for next step simulation
            last_row = current_df.iloc[[-1]].copy()
            if "anomaly_c_lag_1" in last_row.columns:
                last_row["anomaly_c_lag_1"] = next_pred
            current_df = pd.concat([current_df, last_row], ignore_index=True)

        return np.array(preds, dtype=float)

    def save(self, model_dir: Path) -> Path:
        trained_dir = settings.trained_models_dir
        metadata_dir = settings.metadata_dir
        trained_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        weights_path = trained_dir / "lstm_weights.pt"
        scaler_path = trained_dir / "lstm_scaler.joblib"
        meta_path = metadata_dir / "lstm_metadata.json"

        if self.model is not None:
            torch.save(self.model.state_dict(), weights_path)
        joblib.dump(self.scaler, scaler_path)

        meta_data = self.get_metadata()
        meta_path.write_text(json.dumps(meta_data, indent=2), encoding="utf-8")
        logger.info("Saved PyTorch LSTM model artifacts to %s", trained_dir)
        return meta_path

    def load(self, model_dir: Path) -> "PyTorchLSTMClimateModel":
        trained_dir = settings.trained_models_dir
        metadata_dir = settings.metadata_dir

        weights_path = trained_dir / "lstm_weights.pt"
        scaler_path = trained_dir / "lstm_scaler.joblib"
        meta_path = metadata_dir / "lstm_metadata.json"

        if not weights_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(f"LSTM model artifacts missing in {trained_dir}")

        self.scaler = joblib.load(scaler_path)
        if meta_path.exists():
            self.training_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.feature_names = self.training_meta.get("feature_names", [])
            self.params.update(self.training_meta.get("params", {}))

        input_dim = len(self.feature_names) if self.feature_names else 20
        self.model = LSTMModule(
            input_dim=input_dim,
            hidden_size=int(self.params.get("hidden_size", 64)),
            num_layers=int(self.params.get("num_layers", 2)),
            dropout=float(self.params.get("dropout", 0.2)),
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        self._is_fitted = True
        return self

    def get_metadata(self) -> Dict[str, Any]:
        return self.training_meta
