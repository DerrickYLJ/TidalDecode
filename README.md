### TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention

## Abstract
Large language models (LLMs) have driven significant advancements across diverse NLP tasks, with long-context models gaining prominence for handling extended inputs. However, the expanding key-value (KV) cache size required by Transformer architectures intensifies the memory constraints, particularly during the decoding phase, creating a significant bottleneck. Existing sparse attention mechanisms designed to address this bottleneck have two limitations: (1) they often fail to reliably identify the most relevant tokens for attention, and (2) they overlook the spatial coherence of token selection across consecutive Transformer layers, which can lead to performance degradation and substantial overhead in token selection. 

Given such circumstance, we introduce **TidalDecode**, a simple yet effective algorithm and system for fast and accurate LLM decoding through position persistent sparse attention. TidalDecode leverages the spatial coherence of tokens selected by existing sparse attention methods and introduces a few token selection layers that perform full attention to identify the tokens with the highest attention scores, while all other layers perform sparse attention with the pre-selected tokens. This design enables TidalDecode to substantially reduce the overhead of token selection for sparse
attention without sacrificing the quality of the generated results. Evaluation on a diverse set of LLMs and tasks shows that TidalDecode closely matches the generative performance of full attention methods while reducing the LLM decoding latency by up to **2.1x**.

## Installation
1. Clone the submodules
```
git clone https://github.com/DerrickYLJ/TidalDecode.git
git submodule update --init --recursive
```
2. Install dependency libraries
```
conda create -yn tidal python=3.10
conda activate tidal
pip install -e . && pip install flash-attn==2.3.0 --no-build-isolation
python setup.py develop

# Install CMake (with version >= 3.26.4)
conda install cmake

# build libraft
cd kernels/3rdparty/raft
./build.sh libraft
```
3. Build end-to-end operators with PyBind
```
# This will automatically build and link the operators
cd tidal/ops
bash setup.sh
```

## Small Demo
Run example:

```
python examples/run_tidal_llama.py  --top_k 256 --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k
```

## Performance Evaluation
Run Needle-in-the-Haystack:

```
bash experiments/needle_test/run_needle_test.sh
```

Run perplexity:

```
bash experiments/ppl/run_ppl.sh
```

Run LongBench:

```
bash experiments/LongBench/run_longbench.sh
```


## Efficiency Evaluation
Kernels and end-to-end effiency are evaluated on A100 GPU with CUDA version of 12.2.

### End-to-end Efficiency

To reproduce the end-to-end efficiency results in Figure.10, please execute:
```
cd scripts
bash bench_efficiency_e2e.sh
```

## Future Plan
This repo mainly reproduces the results in our [paper](https://arxiv.org/abs/2410.05076). As TidalDecode is flexible in the choice of the token selection layer, we are developing a library to support the efficient deployment of our method with flexible model configurations that suit users' accuracy/efficiency requirements.
- [ ] Llama3 Model Support + GQA
- [ ] Independent top-k selection by head

## Reference
```
@misc{yang2024tidaldecodefastaccuratellm,
      title={TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention}, 
      author={Lijie Yang and Zhihao Zhang and Zhuofu Chen and Zikun Li and Zhihao Jia},
      year={2024},
      eprint={2410.05076},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.05076}, 
}
```

