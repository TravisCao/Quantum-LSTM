import os
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from data_utils import DataModule
from utils import load_config
from trainer import PricePredictor, load_model, predict

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("QLSTM")
    parser.add_argument("-Q", "--n_qubits", default=7, type=int)
    parser.add_argument("-B", "--backend", default="lightning.qubit")
    parser.add_argument("--input_dim", default=5, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--diff_method", default="best")
    parser.add_argument("--depth", default=1, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--DEBUG", default=False, type=bool)
    parser.add_argument("--model_name", default="QLSTM")
    parser.add_argument("--hidden_dim", default=2, type=int)
    # encoding part: "original" | "No-H" | "No-Square" | "arcsin-arccos"
    parser.add_argument("--encoding", default="original")
    parser.add_argument(
        "--four_linear_before_vqc", dest="four_linear_before_vqc", action="store_true"
    )
    parser.add_argument(
        "--no_four_linear_before_vqc",
        dest="four_linear_before_vqc",
        action="store_false",
    )
    parser.add_argument(
        "--combine_linear_after_vqc",
        dest="combine_linear_after_vqc",
        action="store_true",
    )
    parser.add_argument(
        "--no_combine_linear_after_vqc",
        dest="combine_linear_after_vqc",
        action="store_false",
    )
    parser.add_argument("--vqc", default="original")
    parser.add_argument("--only_price", default=False)
    parser.add_argument("--data", default="period1")
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--resume", default=False)
    parser.add_argument("--id", default=None, type=str)
    parser.set_defaults(four_linear_before_vqc=True, combine_linear_after_vqc=False)

    args = parser.parse_args()

    config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    config.update(vars(args))
    config[args.model_name].update(vars(args))

    seed_everything(args.seed, workers=True)

    data_module = DataModule(config=config)
    data_module.setup()
    model = PricePredictor(config)
    if args.DEBUG:
        trainer = Trainer(
            accelerator=config["accelerator"],
            devices=1,
            max_epochs=1,
            limit_train_batches=3,
            limit_test_batches=3,
            # fast_dev_run=True,
        )
        trainer.fit(model, datamodule=data_module)
    elif args.resume:
        run = wandb.init(project=config["model_name"])
        run.config["model_name"] = args.model_name
        run.config["vqc"] = args.vqc
        run.config["encoding"] = args.encoding
        run.config["data"] = args.data
        run.config["four_linear_before_vqc"] = args.four_linear_before_vqc
        run.config["combine_linear_after_vqc"] = args.combine_linear_after_vqc
        model = load_model(
            run=run,
            exp_id=args.id,
            model_name=args.model_name,
            vqc=args.vqc,
            encoding=args.encoding,
            data=args.data,
            four_linear_before_vqc=args.four_linear_before_vqc,
            hidden_dim=args.hidden_dim,
            combine_linear_after_vqc=args.combine_linear_after_vqc,
        )
        predict(
            model,
            period=args.data,
            max_price=config["MAX_PRICE"],
            min_price=config["MIN_PRICE"],
        )
    else:
        early_stop_callback = EarlyStopping(
            monitor="val-loss",
            mode="min",
            min_delta=0.000001,
            patience=5,
            check_finite=True,
            check_on_train_epoch_end=False,
            verbose=True,
        )

        # DDP with this will log multiple times
        wandb_logger = WandbLogger(
            project=config["model_name"], log_model="all", notes="4 linear"
        )
        wandb_logger.watch(model, log="all")

        checkpoint_callback = ModelCheckpoint(
            monitor="val-loss",
            mode="min",
            save_top_k=-1,  # -1 for saving all models
            every_n_epochs=50,
            filename="{epoch}-{val-loss:.5f}",
            verbose=True,
        )

        trainer = Trainer(
            accelerator=config["accelerator"],
            devices=config["devices"],
            strategy=DDPStrategy(
                find_unused_parameters=config[args.model_name]["find_unused_parameters"]
            ),  # speed up DDP
            log_every_n_steps=5,
            deterministic=True,
            check_val_every_n_epoch=10,
            default_root_dir="model_checkpoints",
            callbacks=[early_stop_callback, checkpoint_callback],
            max_epochs=600,
            min_epochs=300,
            logger=wandb_logger,
        )
        trainer.fit(model, datamodule=data_module)

        trainer.test(model, datamodule=data_module)
