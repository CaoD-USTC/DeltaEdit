# DeltaEdit

Code for [On the Superimposed Noise Accumulation Problem in Sequential Knowledge Editing of Large Language Models] (AAAI 2026)

### Train
```
CUDA_VISIBLE_DEVICES=x python3 -m experiments.evaluate --alg_name=DeltaEdit --model_name=gpt2-xl --hparams_fname=gpt2-xl.json --num_edits=1 --dataset_size_limit 3000
```
### Eval
```
python3 -m experiments.summarize --dir_name=DeltaEdit --runs=run_xxx
```
