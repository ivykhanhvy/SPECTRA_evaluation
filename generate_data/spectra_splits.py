import os
import deepchem as dc
from spectrae import Spectra, SpectraDataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tqdm import tqdm
from os.path import join
import pickle
from pathlib import Path
import argparse


class MolnetDataset(SpectraDataset):
  def parse(self, dataset):
    return dataset

  def __len__(self):
    return len(self.samples)

  def sample_to_index(self,sample):
    if not hasattr(self, 'index_to_sequence'):
      print('Generating index to sequence')
      self.index_to_sequence = {}
      for i in tqdm(range(len(self.samples))):
        x = self.__getitem__(i)
        self.index_to_sequence[x] = i
    return self.index_to_sequence[sample]

  def __getitem__(self, idx):
    return self.samples[idx]

class MolnetTanimotoSpectra(Spectra):
  def spectra_properties(self, sample_one, sample_two):
    return TanimotoSimilarity(sample_one, sample_two)

  def cross_split_overlap(self, train, test):
    average_similarity = []
    for i in train:
      for j in test:
        average_similarity.append(self.spectra_properties(i,j))
    return np.mean(average_similarity)

def convert_to_morgan_fingerprint(dataset_name, base_path):
  dataset = pd.read_csv(f'{base_path}/dataset/curated_dataset/{dataset_name}.csv')
  dataset_smiles = dataset['smiles']

  mfp = []
  for i in range(len(dataset_smiles)):
    mol = Chem.MolFromSmiles(dataset_smiles[i])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    mfp.append(fp)

  return mfp

def generate_spectra_tanimoto_splits(dataset_name, spectra_parameters, base_path):
  mfp = convert_to_morgan_fingerprint(dataset_name, base_path)
  save_dir = f'{base_path}/raw_splits/spectra_tanimoto/{dataset_name}'
  os.makedirs(save_dir, exist_ok=True)

  spectra_dataset = MolnetDataset(mfp, f'{dataset_name}')
  tanimoto_spectra = MolnetTanimotoSpectra(spectra_dataset, binary=False)
  tanimoto_spectra.pre_calculate_spectra_properties(f'{dataset_name}', force_recalculate=False)
  tanimoto_spectra.generate_spectra_splits(**spectra_parameters)

  stats = tanimoto_spectra.return_all_split_stats()
  stats_df = pd.DataFrame(stats).sort_values(by='SPECTRA_parameter', ascending=True)

  return stats_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Run SPECTRA Taniomoto Splits')
    parser.add_argument('--dataset_name', type =str, required=True)
    parser.add_argument('--base_path', type=str, required=True)
    args = parser.parse_args()
    spectra_parameters = {'number_repeats': 3,
                          'random_seed': [42, 44, 46],
                          'spectral_parameters': ["{:.2f}".format(i) for i in np.arange(0, 1.05, 0.05)],
                          'force_reconstruct':False}
    generate_spectra_tanimoto_splits(args.dataset_name, spectra_parameters, args.base_path)
    