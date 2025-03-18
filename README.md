<div align="center">


  <img src="media/header_compress.png" width="800" alt="NVIDIA Isaac GR00T N1 Header">
  
  <!-- --- -->
  
  <p style="font-size: 1.2em;">
    <a href="https://developer.nvidia.com/isaac/gr00t"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/collections/nvidia/gr00t-67ab9006952ccb4d1e468e5e"><strong>HuggingFace</strong></a> | 
    <a href="https://placeholder"><strong>Paper</strong></a>
  </p>
</div>

## NVIDIA Isaac GR00T N1

<div align="center">
<video width="800" controls>
  <source src="media/dual-humanoid-industrial.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</div>


NVIDIA Isaac GR00T N1 is the world’s first open foundation model for generalized humanoid robot reasoning and skills. This cross-embodiment model takes multimodal input, including language and images, to perform manipulation tasks in diverse environments.

GR00T N1 was trained on an expansive humanoid dataset, consisting of real captured data, synthetic data generated using the components of NVIDIA Isaac GR00T Blueprint, and internet-scale video data. It is adaptable through post-training for specific embodiments, tasks and environments.

The neural network architecture of GR00T N1 is a combination of vision-language foundation model and diffusion transformer head that denoises continuous actions. Here is a schematic diagram of the architecture:

<div align="center">
<img src="media/model-architecture.png" width="600" alt="model-architecture">
</div>


GR00T N1 is trained on a diverse corpus of data on various hardware platforms and from all kinds of sources, such as human-teleoperated real robot demonstrations, GR00T-Mimic synthetic data from physics simulation, and in-the-wild videos: 

<div align="center">
<img src="media/real-data.gif"  alt="real-robot-data">
<img src="media/sim-data.gif"  alt="sim-robot-data">
</div>

Here is the general procedure to use GR00T N1:

1. Assuming the user has already collected a dataset of robot demonstrations in the form of (video, state, action) triplets. 
2. User will first convert the demonstration data into the LeRobot compatible data schema, which is compatible with the upstream [Huggingface LeRobot](https://github.com/huggingface/lerobot).
3. Our repo provides examples to configure different configurations for training with different robot embodiments.
4. Our repo provides convenient scripts to finetune the pretrained GR00T N1 model on user's data, and run inference.
5. User will connect the `GrootPolicy` to the robot controller to execute actions on their target hardware.

## Target Audience

GR00T N1 is intended for researchers and professionals in humanoid robotics. This repository provides tools to:

- Leverage a pretrained foundation model for robot control
- Fine-tune on small, custom datasets
- Adapt the model to specific robotics tasks with minimal data
- Deploy the model for inference

The focus is on enabling customization of robot behaviors through finetuning.

## Prerequisites
- We have tested the code on Ubuntu 20.04 and 22.04, GPU: H100, L40 and A6000 for finetuning and Python==3.10, CUDA version 12.4. Single A6000 can be used for finetuning.
- For inference, we have tested on Ubuntu 20.04 and 22.04, GPU: 4090, A6000
- Please make sure you have the following dependencies installed in your system: `ffmpeg`, `libsm6`, `libxext6`


## Installation Guide

Clone the repo:

```sh
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

Create a new conda environment and install the dependencies. We recommend Python 3.10:

> Note that, please make sure your CUDA version is 12.4. Otherwise, you may have a hard time with properly configuring flash-attn module.

```sh
conda create -n groot python=3.10
conda activate groot
pip3 install --upgrade setuptools
pip install -e .
pip3 install --no-build-isolation flash-attn==2.7.1.post4 
```

Create a HuggingFace token by going to [this page](https://huggingface.co/settings/tokens). (Skip if you already have a token.)
```sh
huggingface-cli login
```

## Getting started with this repo

We provide accessible Jupyter notebooks and detailed documentations in the [`./getting_started`](./getting_started) folder. Utility scripts can be found in the [`./scripts`](./scripts) folder.

## 1. Data Format & Loading

- To load and process the data, we use [Huggingface LeRobot data](https://github.com/huggingface/lerobot), but with a more detailed metadata and annotation schema (we call it "LeRobot compatible data schema").
- This schema requires data to be formatted in a specific directory structure to be able to load it. 
- Here's an example of the schema is stored here: `./demo_data/robot_sim.PickNPlace` 
```
.
├─meta 
│ ├─episodes.jsonl
│ ├─modality.json
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   ├─observation.images.ego_view_pad_res256_freq20
│   │ └─episode_000002.mp4
│   └─observation.images.egoview
│     └─episode_000002.mp4
└─data
  └─chunk-000
    ├─episode_000001.parquet
    └─episode_000000.parquet
```
- Data organization guide is available in [`getting_started/LeRobot_compatible_data_schema.md`](getting_started/LeRobot_compatible_data_schema.md)
- Once your data is organized in this format, you can load the data using `LeRobotDataLoader` class.

```python
from groot.data.dataset import LeRobotSingleDataset
from groot.data.embodiment_tags import EmbodimentTag
from groot.data.dataset import ModalityConfig
from groot.experiment.data_config import DATA_CONFIG_MAP


# get the data config
data_config = DATA_CONFIG_MAP["gr1_unified"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="demo_data/robot_sim.PickNPlace",
    modality_configs=modality_config,
    embodiment_tag=EmbodimentTag.GR1_UNIFIED, # the embodiment to use
)

# This is an example of how to access the data.
dataset[5]
```

- [`getting_started/0_load_dataset.ipynb`](getting_started/0_load_dataset.ipynb) is an interactive tutorial on how to load the data and process it to interface with the GR00T N1 model.
- [`scripts/load_dataset.py`](scripts/load_dataset.py) is an executable script with the same content as the notebook.
- We provide a script to validate your data against the LeRobot format: [`scripts/validate_lerobot.py`](scripts/validate_lerobot.py).

## 2. Inference

* The GR00T N1 model is hosted on [Huggingface](https://huggingface.co/nvidia/GR00T-1-v0.1.0b)
* Example dataset for a GR1 robot is available as [Huggingface lerobot dataset](https://huggingface.co/datasets/nvidia/gr00t-gr1-apple-to-shelf)

```python
from groot.model.policy import GrootPolicy
from groot.data.embodiment_tags import EmbodimentTag

# 1. Load pre-trained model
policy = GrootPolicy(
    model_path="nvidia/GR00T-N1-2B",
    modality_config=modality_config,
    modality_transform=transforms,
    embodiment_tag=EmbodimentTag.GR1_UNIFIED,
    device="cuda"
)

# 2. Load the dataset
#  we will not load transform in the dataset during inference as we let
# GrootPolicy handles the transforms internally
dataset = LeRobotSingleDataset(.....<Similar to the loading section above>....)

# 3. Run inference
action_chunk = policy.get_action(dataset[0]) # action key is not used during inference
```

- [`getting_started/1_groot_inference.ipynb`](getting_started/1_groot_inference.ipynb) is an interactive Jupyter notebook tutorial to build an inference pipeline.
- [`scripts/groot_inference.py`](scripts/groot_inference.py) is an executable script with the same content as the notebook.

Users can also run the inference service using the provided script. The inference service can be run in server mode or client mode.

```bash
python scripts/inference_service.py --model_path nvidia/GR00T-N1-2B --server
```

On a different terminal, run the client mode to send requests to the server.
```bash
python scripts/inference_service.py  --client
```

## 3. Fine-tuning

User can run the finetuning script below to finetune the model with the example dataset. A tutorial is available in [`getting_started/2_groot_finetuning.ipynb`](getting_started/2_groot_finetuning.ipynb).

Then run the finetuning script:
```bash
# First run --help to see the available arguments
python scripts/groot_finetune.py --help

# then run the script
python scripts/groot_finetune.py --dataset-path ./demo_data/robot_sim.PickNPlace --num-gpus 1
```

You can also download a sample dataset from our huggingface sim data release [here](https://huggingface.co/datasets/nvidia/GR00T-Sim-Post-Training-Data)

```
huggingface-cli download  nvidia/GR00T-Sim-Post-Training-Data \
  --repo-type dataset \
  --include "robocasa_gr1_arms_only_fourier_hands.TwoArmCanSortPadRes256Freq20_30/**" \
  --local-dir $HOME/gr00t_dataset
```

The recommended finetuning configurations is to boost your batchsize to the max, and train for 20k steps.

*Hardware Requirements*
- We used 1 H100 node and L40 node for optimal finetuning. Other hardware configurations (e.g. A6000, RTX4090) would work but will take longer to converge. The exact batchsize is dependent on the hardware, and whether you are tuning which component of the model.

For new emdodiment finetuning, checkout our notebook in [`getting_started/3_new_embodiment_finetuning.ipynb`](getting_started/3_new_embodiment_finetuning.ipynb).

## 4. Evaluation

To conduct an offline evaluation of the model, we provide a script that evaluates the model on a dataset, and plots it out.

Run the newly trained model
```bash
python scripts/inference_service.py --server \
    --model_path <MODEL_PATH> \
    --embodiment_tag new_embodiment
```

Run the offline evaluation script
```bash
python scripts/eval_policy.py --plot \
    --dataset_path <DATASET_PATH> \
    --embodiment_tag new_embodiment
```

You will then see a plot of GT vs Predicted actions, along with unnormed MSE of the actions. This would give you an indication if the policy is performing well on the dataset.


# FAQ

*I have my own data, what should I do next for finetuning?*
- This repo assumes that your data is already organized according to the LeRobot format. 
- You can use `scripts/validate_lerobot.py` to validate your data organization.


*What is Modality Config? Embodiment Tag? and Transform Config?*
- Embodiment Tag: Defines the robot embodiment used, non-pretrained embodiment tags are all considered as new embodiment tags.
- Modality Config: Defines the modalities used in the dataset (e.g. video, state, action)
- Transform Config: Defines the Data Transforms applied to the data during dataloading.
- For more details, see [`getting_started/4_deeper_understanding.md`](getting_started/4_deeper_understanding.md)

*What is the Inference speed for GrootPolicy?*

This number is based on a single L40 GPU. It is approximately the same as on a 4090 GPU.

| Module | Inference Speed |
|----------|------------------|
| VLM Backbone | 22.92 ms |
| Action Head with 4 diffusion steps | 4 x 9.90ms = 39.61 ms |
| Full Model | 62.53 ms |

We noticed with a denoising steps of 4 is sufficient during inference.

# Contributing

**Pre-commit Hooks**

We use pre-commit to manage code formatting and linting. Install and set up pre-commit hooks:

```bash
pip3 install pre-commit
pre-commit install
```

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md)
