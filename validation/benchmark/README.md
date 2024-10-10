## Benchmark tools instructions

### Benchmark data (404 only)
Pre-download the following data:
- Benchmark dataset can be found [here](https://drive.google.com/drive/u/2/folders/1UTF92aXYMjNKfxs5LnP--dPVs5xZNw1J).
- You will need to download all ply files from [here](https://drive.google.com/drive/u/2/folders/1f8r1Gzb01-8rkui_mNW4fohreEJyJHnX).
- You will need to download [this .txt file](https://drive.google.com/drive/u/2/folders/1UTF92aXYMjNKfxs5LnP--dPVs5xZNw1J) with prompts.

### Benchmark data (other users)
To run these tools you will need:
- dataset with prompts for generating 3D objects;
- dataset with generated 3D objects using previously generated prompts;
- evaluation of the generated dataset using manual labeling. The template can be generated using current tool set.

### Before you run:
you will need to provide: 
- path to the ply dataset in "data_folder" field;
- path to the .txt file ("prompts_file") with prompts that were used for generating the dataset (same order as plys were gnerated); 
- path to the .csv template file ("template_path"): 
  - you can provide custom name to generate a template .csv file, e.g. "template.scv". It can be used for manual labeling of the data.
  Then it can be used as a point of reference for validator evaluation;
  - For 404 team: set it to "benchmark_reference.csv";
- to generate template file set "generate_raw_template: True" and "evaluate_validation: False";
- to evaluate validator set "generate_raw_template: False" and "evaluate_validation: True";

### Running tools:
In benchmark tools currently we have two tools:
- benchmark_quality_test_tool.py - this tool generates raw template file and can be used for evaluating the validator quality
if reference template was supplied.
- benchmark_validation_test.py - this tool tests how consistent validation is on the subset of models. You can manually set 
the amount of repetitive iterations for testing in the benchmark_config.yml, e.g "iterations: 30". At the end of the evaluation 
a .csv file with statistics will be generated.

To run one of the tools you just need to do the following from the "benchmark" folder:
```
export PYTHONPATH="${PYTHONPATH}:$(git rev-parse --show-toplevel)/validation"
python benchmark_quality_test_tool.py
python benchmark_validation_test_tool.py
```