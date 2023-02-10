import os

file_path = os.path.dirname(os.path.realpath(__file__))

import candle

additional_definitions = [
    {"name": "classes", "type": int, "default": 2},
    {"name": "label_noise", "type": float},
    {"name": "std_dev", "type": float},
    {"name": "feature_col", "type": int},
    {"name": "sample_ids", "type": int},
    {"name": "feature_threshold", "type": float},
    {"name": "add_noise", "type": candle.str2bool},
    {"name": "noise_correlated", "type": candle.str2bool},
    {"name": "noise_column", "type": candle.str2bool},
    {"name": "noise_cluster", "type": candle.str2bool},
    {"name": "noise_gaussian", "type": candle.str2bool},
    {"name": "noise_type", "type": str},
    {
        "name": "iter-limit", 
        "type": int, 
        "default": 2000,
        "help": "Number of iterations as a limit for benchmarking purpose"
    },
    {
        "name": "gpu-type", 
        "type": str, 
        "default": "unspecified_gpu",
        "help": "GPU type used for benchmarking"
    },
    {
        "name": "num-gpu", 
        "type": int, 
        "default": 4,
        "help": "Number of GPUs used for benchmarking"
    },    
    # {
    #     "name": "conf", 
    #     "type": str, 
    #     "default": "nt3_perf_bench_model.txt",
    #     "help": "benchmarking"
    # },
]

required = [
    "data_url",
    "train_data",
    "test_data",
    "model_name",
    "conv",
    "dense",
    "activation",
    "out_activation",
    "loss",
    "optimizer",
    "metrics",
    "epochs",
    "batch_size",
    "learning_rate",
    "dropout",
    "classes",
    "pool",
    "output_dir",
    "timeout",
]


class BenchmarkNT3(candle.Benchmark):
    def set_locals(self):
        """Functionality to set variables specific for the benchmark
        - required: set of required parameters for the benchmark.
        - additional_definitions: list of dictionaries describing the additional parameters for the
        benchmark.
        """

        if required is not None:
            self.required = set(required)
        if additional_definitions is not None:
            self.additional_definitions = (
                self.additional_definitions + additional_definitions
            )
            # print(self.additional_definitions)
