### Sparse Attention using Index Store
Environment Setup (python version = 3.8):
```
pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```
Run example:
```
python examples/run_index_llama.py  --top_k 256 --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k # top-k retrieval
```
