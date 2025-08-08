import deepchem as dc
import numpy as np
import os
import umap.umap_ as umap
import pickle
from sklearn.cluster import KMeans

def generate_random_and_scaffold_splits(dataset_name, base_path):
    splitter_dic = {"random": dc.splits.RandomSplitter,
                    "scaffold": dc.splits.ScaffoldSplitter}

    featurizer = dc.feat.CircularFingerprint(radius=2, size=1024)
    tasks, molnet_dataset, transformers = getattr(dc.molnet, f'load_{dataset_name}')(featurizer = featurizer, splitter=None, reload=False)
    split_dataset = molnet_dataset[0]
    all_ids = np.array(split_dataset.ids)

    for split_type, splitter_type in splitter_dic.items():
        save_dir = os.path.join(base_path, split_type, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        splitter = splitter_type()

        for index, value in enumerate([42, 43, 44, 45, 46]):
            train, test = splitter.train_test_split(split_dataset, frac_train = 0.8, seed = value)
            train_smiles = set(train.ids)
            test_smiles = set(test.ids)

            train_indices = [index for index, smiles in enumerate(all_ids) if smiles in train_smiles]
            test_indices = [index for index, smiles in enumerate(all_ids) if smiles in test_smiles]

            assert len(set(train_indices) & set(test_indices)) == 0
            assert len(set(train_indices + test_indices)) == len(all_ids)

            print(f"{dataset_name} split {split_type} {index}")

            with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_train_split_{index}.pkl"), "wb") as f:
                pickle.dump(train_indices, f)

            with open(os.path.join(save_dir, f"{dataset_name}_{split_type}_test_split_{index}.pkl"), "wb") as f:
                pickle.dump(test_indices, f)
    return print("Random and scaffold splits done.")

def generate_umap_splits(dataset_name, base_path, n_clusters = 7):
    featurizer = dc.feat.CircularFingerprint(radius=2, size=1024)
    tasks, molnet_dataset, transformers = getattr(dc.molnet, f'load_{dataset_name}')(featurizer=featurizer, splitter=None, reload=False)
    mfp_dataset = molnet_dataset[0].X
    test_size = round(0.2 * len(mfp_dataset), 0)

    umap_save_dir = os.path.join(base_path, f"umap/{dataset_name}")
    os.makedirs(umap_save_dir, exist_ok=True)

    for index, value in enumerate([42, 43, 44, 45, 46]):
        mfp_umap = umap.UMAP(n_components = 2, transform_seed = value).fit_transform(mfp_dataset)
        kmeans = KMeans(n_clusters = n_clusters, random_state = value)
        cluster_labels = kmeans.fit_predict(mfp_umap)

        cluster_index, counts = np.unique(cluster_labels, return_counts=True)
        difference_list = [abs(test_size - i) for i in counts]
        min_cluster_index = difference_list.index(min(difference_list))
        umap_test_indices = np.where(cluster_labels == min_cluster_index)[0]
        umap_train_indices = np.where(cluster_labels != min_cluster_index)[0]

        assert len(set(umap_train_indices) & set(umap_test_indices)) == 0
        assert len(set(np.concatenate([umap_train_indices, umap_test_indices]))) == len(mfp_dataset)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_train_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_train_indices, f)

        with open(os.path.join(umap_save_dir, f"{dataset_name}_umap_test_split_{index}.pkl"), "wb") as f:
                pickle.dump(umap_test_indices, f)
    return print("UMAP splits done.")

if __name__ == "__main__":
    datasets = ['bace_classification', 'tox21', 'bbbp', 'sider', 'clintox', 'hiv', 'delaney', 'freesolv', 'lipo']
    base_path = "/Users/ivymac/Desktop/SAGE_Lab"

    for dataset in datasets:
        generate_random_and_scaffold_splits(dataset, base_path)
        generate_umap_splits(dataset, base_path)