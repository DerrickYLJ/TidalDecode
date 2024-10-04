### TidalDecode

Environment Setup:

## Installation
1. Clone the submodules
```
git submodule update --init --recursive
```
2. Install dependency libraries
```
conda create -yn tidal python=3.10
conda activate tidal
pip install -e . && pip install flash-attn==2.3.0 --no-build-isolation

# Install CMake (with version >= 3.26.4)
conda install cmake

# build libraft
cd kernels/3rdparty/raft
./build.sh libraft
```
3. Build end-to-end operators with PyBind
```
# This will automatically build and link the operators
cd quest/ops
bash setup.sh
```

## Small Demo
Run example:

```
python examples/run_index_llama.py  --top_k 256 --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k
```

## Performance Evaluation
Run Needle-in-the-Haystack:

```
bash examples/needle_test/run_needle_test.sh
```

Run perplexity:

```
bash examples/ppl/run_ppl.sh
```

Run LongBench:

```
bash examples/LongBench/run_longbench.sh
```


## Efficiency Evaluation
Kernels and end-to-end effiency are evaluated on A100 GPU with CUDA version of 12.2.

### End-to-end Efficiency

To reproduce the end-to-end efficiency results in Figure.10, please execute:
```
cd scripts
bash bench_efficiency_e2e.sh
```

