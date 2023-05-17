
# L-QLSTM

by Yuji Cao, Xiyuan Zhou, Xiang Fei, Huan Zhao, Wenxuan Liu and Junhua Zhao.

This repository contains the data and source code used to produce the results presented in:

> "[Title of the Paper and/or Preprint]"

## Abstract

[Abstract of the paper].

### Dependencies

Use requirements.txt to install the dependencies for reproducing the code.

```bash
pip install -r requirements.txt
```

The experimental result is stored by [WandB](https://wandb.ai/site). You need to register your account first. See the quickstart of WandB [here](https://docs.wandb.ai/quickstart).

### Executing program

* `src/train.py` is the main entry for training different models.
* `config.yaml` sets the configuration of data, model and training pipelines.
* `data_utils.py` contains data modules of the dataset including data preprocessing etc.
* `utils.py` includes utility functions.
* `models/qlstm.py` and `models/xx_qlstm.py` implement the quantum-classical models.

```bash
# train QLSTM
python src/train.py --batch_size 16 --model_name QLSTM --devices 16 --accelerator cpu --n_qubits 4 
# train L-QLSTM
python src/train.py --batch_size 16 --model_name xx-QLSTM --devices 16 --accelerator cpu --n_qubits 4 
# train LSTM
python src/run_lstm.py --seed 1 --data period2 --hidden_dim 3 
```

## Data

---

This is the dataset of EU carbon market from 2014.01.01 to 2020.12.31.

### Column Names

* `Price`: carbon price
* `High`: highest price
* `Low`: lowest price
* `Open`: opening price
* `Vol`: trading volume
* `Week`: week number of the year
* `Year`: year of the day
* `t`: remaining days to the last open day of the year

### CSV files

`x_3d.csv` contains features of last day, day before last day, and weekday of last week.

`x_5d.csv` contains features of last five days.

### Period

`period1` contains data during 2014.01.01 - 2016.12.31.

`period2` contains data from 2017.01.01 - 2020.12.31.

## Help

If you have any questions or need further clarification, please feel free to reach out to me at travisyjcao@gmail.com.
