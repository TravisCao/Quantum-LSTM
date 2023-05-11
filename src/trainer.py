import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
from pathlib import Path
from pytorch_lightning import LightningModule
from utils import unscale, count_parameters, load_config
from data_utils import DataModule
from models import XXQLSTM, QLSTM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def load_model(
    run,
    exp_id,
    model_name="xx-QLSTM",
    vqc="original",
    encoding="original",
    data="period1",
    hidden_dim=2,
    four_linear_before_vqc=False,
    combine_linear_after_vqc=False,
):
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    config["model_name"] = model_name
    config[model_name]["vqc"] = vqc
    config[model_name]["encoding"] = encoding
    config[model_name]["hidden_dim"] = hidden_dim
    config[model_name]["four_linear_before_vqc"] = four_linear_before_vqc
    config[model_name]["combine_linear_after_vqc"] = combine_linear_after_vqc
    config["data"] = data
    exp = f"energy-lab/{model_name}/model-{exp_id}:best"
    artifact = run.use_artifact(exp, type="model")
    artifact_dir = artifact.download("../")
    return PricePredictor.load_from_checkpoint(
        Path(artifact_dir) / "model.ckpt", config=config
    )


def predict(model, period, max_price, min_price):
    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    config["data"] = period
    data_module = DataModule(
        config=config, data_dir=os.path.join(os.path.dirname(__file__), "../data/")
    )
    data_module.setup()
    predictions = [[], [], []]
    targets = [[], [], []]
    model.eval()
    for i, dataloader in enumerate(data_module.test_dataloader()):
        for X, y in tqdm(dataloader):
            pred = model(X)
            predictions[i].append(pred.item())
            targets[i].append(y.item())
            if i == 0:
                dataset = "training"
            elif i == 1:
                dataset = "validation"
            else:
                dataset = "test"
            pred_unscale = unscale(
                pred.detach().flatten().numpy(),
                max_price,
                min_price,
            )
            wandb.log({f"pred/{dataset}": pred_unscale})
    mae_fn = lambda a, b: np.abs(np.array(a) - np.array(b)).sum() / len(a)
    mse_fn = lambda a, b: np.square(np.array(a) - np.array(b)).sum() / len(a)
    rmse_fn = lambda a, b: (np.square(np.array(a) - np.array(b)) ** 1 / 2).sum() / len(
        a
    )
    for j, dataset in enumerate(["training", "validation", "test"]):
        wandb.log({f"{dataset}-MSE": mse_fn(predictions[j], targets[j])})
        wandb.log({f"{dataset}-MAE": mae_fn(predictions[j], targets[j])})
        wandb.log({f"{dataset}-RMSE": rmse_fn(predictions[j], targets[j])})
    return predictions


class PricePredictor(LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.model_name = config["model_name"]
        self.n_qubits = config[self.model_name]["n_qubits"]
        self.input_dim = config[self.model_name]["input_dim"]
        self.hidden_dim = config[self.model_name]["hidden_dim"]
        self.depth = config[self.model_name]["depth"]
        self.backend = config["backend"]
        self.target_size = config["target_size"]
        self.n_timestep = config["n_timestep"]
        self.vqc = config[self.model_name]["vqc"]
        self.dropout_rate = config[self.model_name]["dropout"]

        params_to_save = {
            "lr": self.lr,
            "hidden_dim": self.hidden_dim,
            "batch_size": self.batch_size,
            "model_name": self.model_name,
            "n_qubits": self.n_qubits,
            "depth": self.depth,
            "seed": self.config["seed"],
            "device": config["devices"],
            "data": config["data"],
            "vqc": self.vqc,
            "dropout_rate": self.dropout_rate,
        }

        if self.model_name != "LSTM":
            self.diff_method = config[self.model_name]["diff_method"]
            params_to_save["encoding"] = config[self.model_name]["encoding"]

        if self.model_name == "QLSTM":
            self.lstm = QLSTM(config)
        elif self.model_name == "xx-QLSTM":
            self.lstm = XXQLSTM(config)
        else:
            self.lstm = nn.LSTM(
                input_size=self.input_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.depth,
                dropout=self.dropout_rate,
            )

        if self.model_name != "QLSTM":
            self.final = nn.Linear(self.hidden_dim, self.target_size)

        params_to_save["n_parameters"] = count_parameters(self)
        self.save_hyperparameters(params_to_save)

    def forward(self, sequence):
        sequence = (
            sequence.view(-1, self.n_timestep, self.input_dim)
            .transpose(0, 1)
            .contiguous()
        )
        if self.model_name == "QLSTM":
            return self.lstm(sequence)
        lstm_out, _ = self.lstm(sequence)
        return self.final(lstm_out[-1].view(-1, self.hidden_dim))

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).flatten()
        loss = F.mse_loss(pred, y.flatten())
        self.log("train-loss", loss.item(), prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred": pred, "target": y}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x).flatten()
        loss = F.mse_loss(pred, y.flatten())
        self.log("val-loss", loss.item(), prog_bar=True, on_epoch=True)
        return {"loss": loss, "pred": pred, "target": y}

    def test_step(self, batch, batch_idx, dataset_idx):
        part_index_map = {0: "train", 1: "val", 2: "test"}
        part = part_index_map[dataset_idx]
        x, y = batch
        pred = self(x)
        loss = F.mse_loss(pred, y)
        pred_unscale = unscale(
            pred.flatten().numpy(),
            self.config["MAX_PRICE"],
            self.config["MIN_PRICE"],
        )
        target_unscale = unscale(
            y.flatten().numpy(),
            self.config["MAX_PRICE"],
            self.config["MIN_PRICE"],
        )
        self.log("pred", pred_unscale[0], on_step=True, on_epoch=False)
        self.log("target", target_unscale[0], on_step=True, on_epoch=False)
        self.log("test-loss", loss.item(), prog_bar=True)
        return {"loss": loss, "pred": pred, "target": y}

    def predict_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        pred = self(x)
        return {"pred": pred, "target": y}

    def configure_optimizers(self):
        optimizer = (
            torch.optim.Adam(self.parameters(), lr=self.lr)
            if self.config["model_name"] != "QLSTM"
            else torch.optim.RMSprop(
                self.parameters(),
                lr=0.1,
                alpha=0.99,
                eps=1e-8,
            )
        )
        return optimizer
