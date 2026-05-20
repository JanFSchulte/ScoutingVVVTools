"""Shared model loading and inference helpers for BDT and PyTorch NN modes."""

import os
import pickle

import numpy as np
import xgboost as xgb


_EPS = 1e-12


class TorchModelHandle:
    """Small wrapper around a PyTorch module plus its inference metadata."""

    def __init__(self, model, device, feature_names, num_classes, checkpoint):
        self.model = model
        self.device = device
        self.feature_names = list(feature_names)
        self.num_classes = int(num_classes)
        self.checkpoint = dict(checkpoint)


def model_type_from_config(cfg):
    value = cfg.get("model_type", cfg.get("training_mode", "bdt"))
    value = str(value).strip().lower()
    if value in {"bdt", "xgb", "xgboost"}:
        return "bdt"
    if value in {"nn", "neural_network", "neural-network", "mlp", "pytorch"}:
        return "nn"
    raise ValueError(f"Unsupported model_type {value!r}; expected 'bdt' or 'nn'")


def import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "model_type='nn' requires PyTorch. Install a PyTorch build that matches "
            "this machine's CPU/GPU setup before running NN training or inference."
        ) from exc
    return torch, nn, F


def build_torch_mlp(nn, input_dim, hidden_layers, output_dim, activation="silu",
                    dropout=0.0, batch_norm=True):
    activation_key = str(activation).strip().lower()
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
    }
    if activation_key not in activation_map:
        raise ValueError(
            f"Unsupported NN activation {activation!r}; "
            f"expected one of {sorted(activation_map)}"
        )

    layers = []
    prev_dim = int(input_dim)
    for width in hidden_layers:
        width = int(width)
        if width <= 0:
            raise ValueError(f"NN hidden layer widths must be positive, got {width}")
        layers.append(nn.Linear(prev_dim, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(activation_map[activation_key]())
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        prev_dim = width
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _reshape_multiclass_margin(predt, num_class, n_rows=None):
    predt = np.asarray(predt, dtype=float)
    if predt.ndim == 2:
        if predt.shape[1] == num_class:
            return predt
        if predt.shape[0] == num_class:
            return predt.T
    if n_rows is None:
        n_rows = predt.size // num_class
    return predt.reshape(int(n_rows), int(num_class))


def softmax_rows(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    return exp_v / (np.sum(exp_v, axis=1, keepdims=True) + _EPS)


def _torch_load(path, device):
    torch, _, _ = import_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_torch_model(path, num_classes, device=None):
    torch, nn, _ = import_torch()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _torch_load(path, device)
    if not isinstance(checkpoint, dict) or checkpoint.get("model_type") != "nn":
        raise RuntimeError(f"Invalid NN checkpoint: {path}")

    feature_names = checkpoint.get("feature_names")
    if not feature_names:
        raise RuntimeError(f"NN checkpoint missing feature_names: {path}")
    if int(checkpoint.get("num_classes", num_classes)) != int(num_classes):
        raise RuntimeError(
            f"NN checkpoint class count mismatch: checkpoint={checkpoint.get('num_classes')}, "
            f"expected={num_classes}"
        )

    model = build_torch_mlp(
        nn,
        input_dim=int(checkpoint.get("input_dim", len(feature_names))),
        hidden_layers=checkpoint.get("hidden_layers", []),
        output_dim=int(num_classes),
        activation=checkpoint.get("activation", "silu"),
        dropout=float(checkpoint.get("dropout", 0.0)),
        batch_norm=bool(checkpoint.get("batch_norm", True)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return TorchModelHandle(model, device, feature_names, num_classes, checkpoint)


def load_model(model_base, cfg, num_classes, log_message=None):
    configured_type = model_type_from_config(cfg)
    candidates = []
    if configured_type == "nn":
        candidates.append(("nn", model_base + ".pt"))
    else:
        candidates.extend([
            ("bdt_json", model_base + ".json"),
            ("bdt_pkl", model_base + ".pkl"),
        ])

    for kind, path in candidates:
        if not os.path.exists(path):
            continue
        if kind == "nn":
            model = load_torch_model(path, num_classes)
            if log_message:
                log_message(f"Loaded NN model: {path}")
            return model
        if kind == "bdt_json":
            model = xgb.Booster()
            model.load_model(path)
            if log_message:
                log_message(f"Loaded BDT model: {path}")
            return model
        with open(path, "rb") as handle:
            model = pickle.load(handle)
        if log_message:
            log_message(f"Loaded BDT model: {path}")
        return model

    suffix = ".pt" if configured_type == "nn" else "(.json/.pkl)"
    raise FileNotFoundError(f"No {configured_type} model found at {model_base}{suffix}")


def predict_model_logits(model, X, num_classes, batch_size=65536):
    if isinstance(model, TorchModelHandle):
        torch, _, _ = import_torch()
        arr = X.to_numpy(dtype=np.float32, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, arr.shape[0], int(batch_size)):
                batch = torch.as_tensor(arr[start:start + int(batch_size)], dtype=torch.float32, device=model.device)
                parts.append(model.model(batch).detach().cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, int(num_classes)), dtype=float)

    if isinstance(model, xgb.Booster):
        dmat = xgb.DMatrix(X, feature_names=list(X.columns) if hasattr(X, "columns") else None)
        margins = model.predict(dmat, output_margin=True)
        return _reshape_multiclass_margin(margins, num_classes, len(X))

    if hasattr(model, "predict_proba"):
        proba = np.clip(np.asarray(model.predict_proba(X), dtype=float), _EPS, None)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return np.log(proba)
    raise TypeError(f"Unsupported model object for prediction: {type(model)}")


def predict_model_proba(model, X, num_classes, batch_size=65536):
    if isinstance(model, TorchModelHandle):
        return softmax_rows(predict_model_logits(model, X, num_classes, batch_size=batch_size))
    if isinstance(model, xgb.Booster):
        return softmax_rows(predict_model_logits(model, X, num_classes, batch_size=batch_size))
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X), dtype=float)
        proba = np.clip(proba, _EPS, None)
        return proba / np.sum(proba, axis=1, keepdims=True)
    raise TypeError(f"Unsupported model object for prediction: {type(model)}")

