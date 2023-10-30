# CANDLE HPC Benchmarks

This repository is a fork from the [CANDLE](https://github.com/ECP-CANDLE/Benchmarks).

The benchmark is used for our SC'23 paper Toward Sustainable HPC: Carbon Footprint Estimation and Environmental Implications of HPC Systems. Please refer to https://github.com/boringlee24/sc23-sustainability for more information.

There are 5 models that are set up for benchmarking, you can find them in the ```Pilot1/``` directory.

The 5 models are ```Combo```, ```NT3```, ```P1B2```, ```ST1```, and ```TC1```. 

## How to set up

To run in local cluster, create an anaconda environment using the ```environment.yml``` file. Then activate the ``tf2`` conda environment.

To run in AWS, go to the [containerized_distributed_training](https://github.com/boringlee24/containerized_distributed_training) repo and launch the container. Note that in AWS, if you use ``g4dn.xlarge`` instance you may run into out-of-memory issue, consider using ``g4dn.2xlarge`` when debugging on a single T4 GPU.

## How to run in local cluster

Go to the directory of each model. Run this command to benchmark over 4 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python <model-spacific-name>_baseline_keras2.py --gpu-type XXX --num-gpu 4 
```

If there are less number of GPUs (say 2), need to change
```
CUDA_VISIBLE_DEVICES=0,1
```
and inside the ```<model-spacific-name>_default_model.txt``` file, change the ```batch_size``` setting linearly to number of GPUs (e.g., if the value was 256, change it to 128 when reducing GPU number from 4 to 2, or change to 64 if GPU changes from 4 to 1.

## Benchmark data

The collected benchmark data is available in ``benchmark_logs`` directory. For example, 4xv100 represents running over 4 V100 GPUs, carbon_{testcase}.json reports the operational carbon, while time_{testcase}.json reports the mini-batch time, representing performance.

---------------------------------

# Benchmarks

ECP-CANDLE Benchmarks

This repository contains the CANDLE benchmark codes. These codes implement deep learning architectures that are relevant to problems in cancer. These architectures address problems at different biological scales, specifically problems at the molecular, cellular and population scales.

The naming conventions adopted reflect the different biological scales.

Pilot1 (P1) benchmarks are formed out of problems and data at the cellular level. The high level goal of the problem behind the P1 benchmarks is to predict drug response based on molecular features of tumor cells and drug descriptors.

Pilot2 (P2) benchmarks are formed out of problems and data at the molecular level. The high level goal of the problem behind the P2 benchmarks is molecular dynamic simulations of proteins involved in cancer, specifically the RAS protein.

Pilot3 (P3) benchmarks are formed out of problems and data at the population level. The high level goal of the problem behind the P3 benchmarks is to predict cancer recurrence in patients based on patient related data.

Each of the problems (P1,P2,P3) informed the implementation of specific benchmarks, so P1B3 would be benchmark three of problem 1.
At this point, we will refer to a benchmark by it's problem area and benchmark number. So it's natural to talk of the P1B1 benchmark. Inside each benchmark directory, there exists a readme file that contains an overview of the benchmark, a description of the data and expected outcomes along with instructions for running the benchmark code.

Over time, we will be adding implementations that make use of different tensor frameworks. The primary (baseline) benchmarks are implemented using keras, and are named with '\_baseline' in the name, for example p3b1_baseline_keras2.py.

Implementations that use alternative tensor frameworks, such as pytorch, will have the name of the framework in the name. Examples can be seen in the Pilot1 benchmark UnoMT directory, for example: `unoMT_baseline_pytorch.py`

Documentation: https://ecp-candle.github.io/Candle/
