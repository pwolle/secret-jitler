# Secret-Jitler
A **j**ust-**i**n-**t**ime compiled python implementation of the popular parlor game [Secret Hitler](https://www.secrethitler.com/assets/Secret_Hitler_Rules.pdf).

## Features
- ⚡ blazingly fast: gpu/tpu-compatible and arbitrarily parallelizeable
- 🤖 BYOB: Build your own bot 
- 🛠️ comprehensive API for training bots
- 🧠 play with a variety of AI-competitors
- 🤯 see what's going on thanks to the narration 

## Motivation
As avid players of the game Secret Hitler, we are always discussing our strategies. Since we are programmmers, we decided to create a test for these strategies in simulated games.

## Installation

### System requirements and download
1. Make sure python 3.10 or newer is installed on your system, if you are not sure about your currently installed version, run `python --version`.
2. Clone the repository with `git clone https://github.com/unitdeterminant/secret-jitler.git`, if `git` is not available download directly from [github](https://github.com/unitdeterminant/secret-jitler.git) and unzip it.
3. Open a terminal in `secret-jitler`
4. Make sure [pip](https://pip.pypa.io/en/stable/installation/) is installed and run `pip install -r requirements.txt` 

Tested on Ubuntu 22.04 (LTS), Debian 11.6, MX-21.3, [gwdg jupyter](https://jupyter-cloud.gwdg.de) and macOS Ventrua 13.1.

### Running on GPU via Conda
If you want to train a bot or run lots of games in parallel using a gpu is recommended. You can check, whether your gpu is cuda compatile [here](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Open a terminal in `secret-jitler`
3. run `conda env create -f environment.yml`
4. activate the environment via `conda activate jaxgpu`

Tested on Manjaro 22.0.0 with a gtx 1060.

### First steps
If you want to play an interactive game, open a terminal, navigate to the `project` directory and run `python play.py`.
You might even use some command line options (run `python play.py -h` for a list of available options) for more control.


## API reference
If you want to build a bot, take a look at the [documentation](https://github.com/unitdeterminant/secret-jitler/blob/main/project/bots/README.md) in `project/game/bots/README.md`.


## Performance 🥵
Performance numbers estimated using the script `project/performance.py`.
| hardware | throughput in it/s | batch size |
| - | - | - |
| gtx 1060 | 2.9e6 | 131072 |
| i7-6700  | 5.8e4 | 256 |
| AMD 7 4700U | 7.1e4 | 128 |
| [gwdg jupyter](https://jupyter-cloud.gwdg.de) | 3.9e4 | 128 |

## Tests
This repo includes a `test.py` file to validate the game states. User inputs are always validated.


## Contributions
- __Uncreative-Nickanme__ executive session, documentation
- __OhKjell__ testing, narration, interactive
- __JannisRow__ elective session, documentation
- __unitdeterminant__ legislative session, refactoring, performance
