import deepchem as dc
from chemprop import featurizers, data, nn, models
import pandas as pd
import numpy as np
import pickle
from lightning import pytorch as pl
from os.path import join
from pathlib import Path

def chemprop(dataset_name, split_type, base_path, task_type):
    #Load Dataset
    df = pd.read_csv(f'{base_path}/dataset/{dataset_name}.csv')
    smis = df.loc[:, 'smiles'].values
    ys = df.drop(columns='smiles').values
    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, ys)]

    #Define Chemprop Model
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    num_tasks = len(df.columns.drop('smiles'))
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    batch_norm = False

    if task_type == 'classification':
        ffn = nn.BinaryClassificationFFN(n_tasks = num_tasks)
        metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score(), nn.metrics.BinaryAccuracy(),nn.metrics.BinaryAUPRC()]
    elif task_type == 'regression':
        ffn = nn.RegressionFFN()
        metric_list = [nn.metrics.RMSE(),nn.metrics.MAE(),nn.metrics.R2Score()]
    else:
        raise ValueError(f"task_type must be either 'classification' or 'regression'")

    #Save Directory
    dir = f'metrics/chemprop/{split_type}'
    output = Path(base_path) / dir
    output.mkdir(parents = True, exist_ok = True)
    filename = f'{dataset_name}_{split_type}_metrics.csv'
    csv_path = join(base_path, dir, filename)

    def train(train_indices, test_indices):
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            accelerator="auto",
            max_epochs=20)

        train_dset = data.MoleculeDataset([all_data[j] for j in train_indices], featurizer=featurizer)
        test_dset = data.MoleculeDataset([all_data[j] for j in test_indices], featurizer=featurizer)

        y_train = np.array([train_dset[j].y for j in range(len(train_dset))])
        y_test = np.array([test_dset[j].y for j in range(len(test_dset))])
        print(f'Train size: {len(train_indices)}')
        print(f'Test size: {len(test_indices)}')

        if ffn == nn.BinaryClassificationFFN():
            assert np.any(y_train == 0) and np.any(y_train == 1)
            assert np.any(y_test == 0) and np.any(y_test == 1)

        train_loader = data.build_dataloader(train_dset)
        test_loader = data.build_dataloader(test_dset)

        mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)
        trainer.fit(mpnn, train_loader)
        results = trainer.test(mpnn, test_loader)

        return results

    def metrics_to_csv(results, row_name):
        row = pd.DataFrame(results, index= [row_name])
        if Path(csv_path).exists():
            df = pd.read_csv(csv_path, index_col=0)
            col_order = df.columns.tolist()
            row = row[col_order]
            df = pd.concat([df, row], axis=0)
        else:
            df = row
        df.to_csv(csv_path)

    #Training
    if split_type in ['spectra_tanimoto','spectra_hamming']:
        root = Path(base_path) / "splits" / split_type / f"{dataset_name}_SPECTRA_splits"
        for parameter in range(8,21):
            parameter = f'{parameter/20:.2f}'
            for i in range(3):
                sp_dir = root / f"SP_{parameter}_{i}"
                train_file = sp_dir / "train.pkl"
                test_file = sp_dir / "test.pkl"
                stats_file = sp_dir / "stats.pkl"

                if not (train_file.exists() and test_file.exists() and stats_file.exists()):
                    continue
                with train_file.open("rb") as f:
                    train_indices = pickle.load(f)
                with test_file.open("rb") as f:
                    test_indices = pickle.load(f)
                with stats_file.open("rb") as f:
                    stat_info = pickle.load(f)

                results = train(train_indices, test_indices)
                merge = {**stat_info, **results[0]}
                metrics_to_csv(merge, f'{dataset_name}_{split_type}_{parameter}_{i}')
                print(f'Done with {dataset_name}_{split_type}_{parameter}_{i}')
        print(f'Complete {dataset_name}_{split_type}')
    else:
        for i in range(5):
            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_train_split_{i}.pkl'),
                      'rb') as f:
                train_indices = pickle.load(f)

            with open(join(base_path,
                           f'splits/{split_type}/{dataset_name}/{dataset_name}_{split_type}_test_split_{i}.pkl'),
                      'rb') as f:
                test_indices = pickle.load(f)

            results = train(train_indices, test_indices)

            metrics_to_csv(results, f'{dataset_name}_{split_type}_{i}')
            print(f'Done with {dataset_name}_{split_type}_{i}')
        print(f'Complete with {dataset_name}_{split_type}')
    return csv_path

if __name__ == '__main__':
    classification_datasets = []
    regression_datasets = ['lipo']
    split_types = ['spectra_hamming']
    for dataset in regression_datasets:
        for split_type in split_types:
            chemprop(dataset,split_type,'/Users/ivymac/Desktop/SAGE_Lab/','regression')