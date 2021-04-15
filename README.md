Code for paper: [Systematic Generalization on gSCAN with Language Conditioned Embedding](https://arxiv.org/pdf/2009.05552.pdf)


# Data Preprocessing 

First clone https://github.com/LauraRuis/multimodal_seq2seq_gSCAN , enter `multimodal_seq2seq_gSCAN/read_gscan` and run
```
python read_gscan.py --dataset_path=../data/compositional_splits/dataset.txt --save_data --output_file=parsed_dataset.txt
```
Then move `parsed_dataset.txt` to `parsed_dataset/` in this repo and run `preprocess_parsed_dataset.py`.

# Environment

We use the same environment as the [baseline](https://github.com/LauraRuis/multimodal_seq2seq_gSCAN), plus [dgl library](https://www.dgl.ai/pages/start.html).

# Training
``
python main_model.py
``

## Parameters
Model parameters are defined in `model/config.py`, but we support quick setting through command line.
- `--run exp_name` Set the experiment name
- `--txt` If enabled, the model redirects all of the outputs to exp/exp_name.txt
- `--load path_to_model` Load the checkpoint
- `--baseline` Switch the model to baseline

# Model Comparision
Run two models on same data splits and compare their results. Results are saved in json format that is compatible with the visualization code from gSCAN.

Note: Modify `model_compare.py` to define the path to models' checkpoints.
``
python model_compare.py
``

# Model Evaluation (depreciated)
In `eval_best_model.py`.

