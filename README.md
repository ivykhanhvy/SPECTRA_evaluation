# **Chemical space separation: Scaffold and UMAP splits may be closer to random than we think**

This repository presents a research framework for evaluating chemical data splitting strategies in machine learning and deep learning. We compared state-of-the-art methods including random, scaffold, and UMAP splits by analyzing train–test overlap and its effect on model performance and generalization.

## **List of subdirectories**
- `data_analysis`: code and results analyzing the results + generating plots
- `datasets`: Contains the MoleculeNet datasets, before and after curation
- `generate_data`: code for curating data, generating random, scaffold, UMAP, 
  and SPECTRA splits, calculating cross-split overlap, and organizing split 
  data for model training, including a `cluster_scripts` subdirectory which 
  has shell files for running HPC/cluster jobs
- `model_info`: Contains any extra information relevant to a model, in 
  particular contains the outcome of the chemprop hyperparameter 
  optimization searches for each of the datasets
- `plots`: All figures for the paper + supplement
- `results`: All results for all models
- `splits_data`: Includes raw splits (.pkl files of 8:2 train:test splits) and .
  json files of 7:1:2 train:val:test random, scaffold, 
  UMAP and SPECTRA splits


## **Installation** 
The required packages and their versions are provided in `requirements.txt`.  

```bash
pip install -r requirements.txt
```

## **Instructions for running the pipeline**
### **1. Generate splits**
Data in `datasets` was curated to remove invalid SMILES structure and 
replicates using `generate_data/data_curation.py`. Next, random, scaffold, and 
UMAP splits were generated using `generate_data/splits.py` and SPECTRA splits were 
generated using `generate_data/spectra_splits.py` to produce 8:2 train:test splits. 
All raw splits were stored as `.pkl` files in `raw_splits`. Next, raw splits 
were converted to index-based splits with 7:1:2 train:val:test sets stored 
as `.json` files using `generate_data/chemprop_data.py`. Final index-based splits 
were stored in `splits_data/hpopt` and `splits_data/chemprop_data` for 
hyperparameter optimization and model trainin, respectively. Cross-split 
overlaps of all four splitting strategies was calculated during the 
execution of `generate_data/cross_split_overlap.py` by taking pairwise Tanimoto 
Similarity between 7:2 train and test sets (excluding validation set) and 
stored in `splits_data/cross_split_overlap`. 


### **2. Train classical models**

Classical models were trained using the code in 
`generate_data/classical_model_baselines.py`. The results were stored in 
`results/classical_results` and analyzed using scripts in the 
`data_analysis` folder.

### **3. Train Chemprop models**

Chemprop hyperparameter optimization was performed using 
`generate_data/cluster_scripts/run_chemprop_hpopt.sh`. The `best_config.toml` 
files for each dataset stored in `model_info/chemprop_hpopt_config` were 
then applied to train chemprop models across all four splitting strategies  
using `generate_data/cluster_scripts/run_chemprop_train_classification.sh` 
and `generate_data/cluster_scripts/run_chemprop_train_regression.sh`. 
Chemprop .log files containing metrics (not included in the repository due 
to size limit) were extracted using `generate_data/extract_chemprop_log.py`. 
All results were recorded in `results/chemprop_results` and statistical  
analysis was performed using `data_analysis/stat_significance_chemprop.R`.
