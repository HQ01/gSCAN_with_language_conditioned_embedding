# Data Preprocessing 
First clone https://github.com/LauraRuis/multimodal_seq2seq_gSCAN , enter read_gscan and run
```
python read_gscan.py --dataset_path=../data/compositional_splits/dataset.txt --save_data --output_file=parsed_dataset.txt
```
Then use our `preprocess_parsed_dataset.py` to preprocess `parsed_dataset.txt` for second time.

# Train Full Model
``
python main_model.py
``

## Parameters
Model parameters are defined in `model/config.py`, but we support quick setting through command line.
- `--run run_name` Define the run name
- `--txt` If enabled, redirect all output to exp/run_name.txt
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

