# SW-MCTS

Decentralized Coordination for Multi-Agent Data Collection in Dynamic Environments (IEEE Transactions on Mobile Computing)

This project contains the source code for the environment constructor, planning algorithms, and simulations used for the IEEE Transactions on Mobile Computing paper listed above, implemented in the Python programming language. The repository contains several folders:

## Installation
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements libraries.
```bash
pip install -r requirements.txt
```

Or use the [conda](https://docs.conda.io/projects/conda/en/stable/) to create a testing environment.
```bash
conda create --name <env> --file requirements.txt
```

## Usage
To run a simulation of SW-MCTS on a multi-agent path planning task.
```bash
python3 sw_mcts_simulation.py [-h] [-s] -m {Dec,Global,Reset,SW} [-f FOLDER] [-v] [-p PARAMS [PARAMS ...]]
optional arguments:
  -h, --help            show this help message and exit
  -s, --save            Save performance
  -m {Dec,Global,Reset,SW}, --mode {Dec,Global,Reset,SW}
                        Type of adaptation to environmental changes
  -f FOLDER, --folder FOLDER
                        Folder name to store simulation data
  -v, --verbose         Print details
  -p PARAMS [PARAMS ...], --params PARAMS [PARAMS ...]
                        Parameter testing

```

To construct new environment configurations.
```bash
python3 graph_helper.py [-h] [-a] [-n N_CONFIGS] [-d DRAW]

Graph constructor

optional arguments:
  -h, --help            show this help message and exit
  -a, --animation       Show constructed graph
  -n N_CONFIGS, --n_configs N_CONFIGS
                        No of configurations
  -d DRAW, --draw DRAW  Draw existing graph
```

To cite this work: 
@article{nguyen2024decentralized,
  title={Decentralized Coordination for Multi-Agent Data Collection in Dynamic Environments},
  author={Nguyen, Nhat and Nguyen, Duong and Kim, Junae and Rizzo, Gianluca and Nguyen, Hung},
  journal={IEEE Transactions on Mobile Computing},
  year={2024},
  publisher={IEEE}
}
```
