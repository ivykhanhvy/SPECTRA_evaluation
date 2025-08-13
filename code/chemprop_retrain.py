import deepchem as dc
from chemprop import featurizers, data, nn, models
import pandas as pd
import numpy as np
import pickle
from lightning import pytorch as pl
from os.path import join
from pathlib import Path

def chemprop_random_scaffold_umap(dataset_name, split_type, path):
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    batch_norm = False

    if dataset_name in {'bace_classification','bbbp','hiv'}:
        ffn = nn.BinaryClassificationFFN()
        metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score(), nn.metrics.BinaryAccuracy(),nn.metrics.BinaryAUPRC()]
    elif dataset_name in {'sider','clintox','tox21'}:
        ffn = nn.MulticlassClassificationFFN()
        metric_list = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score(), nn.metrics.BinaryAccuracy()]
    elif dataset_name in {'lipo','delaney','freesolv'}:
        ffn = nn.RegressionFFN()
        metric_list = [nn.metrics.RMSE(),nn.metrics.MAE(),nn.metrics.R2Score()]

    tasks, molnet_dataset, transformers = getattr(dc.molnet, f'load_{dataset_name}')(splitter=None, reload=False)
    smiles = molnet_dataset[0].ids
    labels = molnet_dataset[0].y

    all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smiles, labels)]
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="auto",
        max_epochs=20)

    dir = f'chemprop/{split_type}'
    output = Path(path) / dir
    output.mkdir(parents=True, exist_ok=True)
    filename = f'{dataset_name}_{split_type}_metrics.csv'
    csv_path = join(path, dir, filename)

    for i in range(5):
        with open(join(path,f'data/{split_type}/{dataset_name}/{dataset_name}_{split_type}_train_split_{i}.pkl'), 'rb') as f:
            train_indices = pickle.load(f)

        with open(join(path, f'data/{split_type}/{dataset_name}/{dataset_name}_{split_type}_test_split_{i}.pkl'), 'rb') as f:
            test_indices = pickle.load(f)

        train_dset = data.MoleculeDataset([all_data[j] for j in train_indices], featurizer = featurizer)
        test_dset = data.MoleculeDataset([all_data[j] for j in test_indices], featurizer = featurizer)

        y_train = np.array([train_dset[j].y for j in range(len(train_dset))])
        y_test = np.array([test_dset[j].y for j in range(len(test_dset))])
        if ffn == nn.BinaryClassificationFFN():
            assert np.any(y_train == 0) and np.any(y_train == 1)
            assert np.any(y_test == 0) and np.any(y_test == 1)

        train_loader = data.build_dataloader(train_dset)
        test_loader = data.build_dataloader(test_dset)

        mpnn = models.MPNN(mp,agg,ffn,batch_norm, metric_list)
        trainer.fit(mpnn, train_loader)
        results = trainer.test(mpnn, test_loader)

        row = pd.DataFrame(results,index = [f'{dataset_name}_{split_type}_{i}'])

        if Path(csv_path).exists():
            df = pd.read_csv(csv_path, index_col = 0)
            row = row[col_order]
            df = pd.concat([df,row], axis = 0)
        else:
            df = row
            col_order = df.columns.tolist()
        df.to_csv(csv_path)
    return csv_path

if __name__ == '__main__':
    datasets = ['hiv']
    split_types = ['umap']
    for dataset in datasets:
        for split_type in split_types:
            chemprop_random_scaffold_umap(dataset,split_type,'/Users/ivymac/Desktop/SAGE_Lab/')