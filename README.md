# Linear-Layer-Enhanced Quantum Long Short-Term Memory for Carbon Price Forcasting

<p align="center">
  <a href="http://dx.doi.org/10.1007/s42484-023-00115-2" alt="DOI">
    <img src="https://zenodo.org/badge/DOI/10.1007/s42484-023-00115-2.svg" /></a>
  <img src ="https://img.shields.io/badge/-Quantum Machine Intelligence-green"/>
  <a href="https://www.python.org/downloads/release/python-380/" alt="Python 3.8">
    <img src="https://img.shields.io/badge/python-3.8-red.svg" /></a>
</p>


The official source code for [**Linear-Layer-Enhanced Quantum Long Short-Term Memory for Carbon Price Forecasting**](http://dx.doi.org/10.1007/s42484-023-00115-2), accepted at Quantum Machine Intelligence (July 2023).

<p align="center">
  <img width="85%" src="img/L-QLSTM.png">
</p>

This code implements the quantum LSTM for price forecasting problems. By changing the data, you can also execute it for other kinds of forecasting or regression problems.

## Cite the paper

```
@article{cao2023linear,
  title={Linear-layer-enhanced quantum long short-term memory for carbon price forecasting},
  author={Cao, Yuji and Zhou, Xiyuan and Fei, Xiang and Zhao, Huan and Liu, Wenxuan and Zhao, Junhua},
  journal={Quantum Machine Intelligence},
  volume={5},
  number={2},
  pages={26},
  year={2023},
  publisher={Springer}
}
```

### Dependencies

Use requirements.txt to install the dependencies for reproducing the code.

```bash
pip install -r requirement.txt
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
