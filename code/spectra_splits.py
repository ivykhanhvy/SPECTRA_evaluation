import deepchem as dc
from spectrae import Spectra, SpectraDataset
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from tqdm import tqdm

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

class MolnetHammingSpectra(Spectra):
  def spectra_properties(self, sample_one, sample_two):
      return np.sum(sample_one != sample_two)/1024

  def cross_split_overlap(self, train, test):
    average_similarity = []
    for i in train:
      for j in test:
        average_similarity.append(self.spectra_properties(i,j))
    return np.mean(average_similarity)

def generate_spectra_tanimoto_splits(dataset_name, spectra_parameters):
  tasks, molnet_dataset, transformers = getattr(dc.molnet, f'load_{dataset_name}')(splitter=None,reload=False)
  dataset_smiles = molnet_dataset[0].ids

  mfp = []
  for i in range(len(dataset_smiles)):
    mol = Chem.MolFromSmiles(dataset_smiles[i])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    mfp.append(fp)

  spectra_dataset = MolnetDataset(mfp, f'{dataset_name}')
  tanimoto_spectra = MolnetTanimotoSpectra(spectra_dataset, binary=False)
  tanimoto_spectra.pre_calculate_spectra_properties(f'{dataset_name}', force_recalculate = False)
  tanimoto_spectra.generate_spectra_splits(**spectra_parameters)

  stats = tanimoto_spectra.return_all_split_stats()
  stats_df = pd.DataFrame(stats).sort_values(by='SPECTRA_parameter', ascending=True)

  return stats_df

def generate_spectra_hamming_splits(dataset_name, spectra_parameters):
  tasks, molnet_dataset, transformers = getattr(dc.molnet, f'load_{dataset_name}')(splitter=None, reload=False)
  dataset_smiles = molnet_dataset[0].ids

  mfp = []
  for i in range(len(dataset_smiles)):
    mol = Chem.MolFromSmiles(dataset_smiles[i])
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024).GetFingerprint(mol)
    mfp.append(fp)
  mfp = np.array(mfp)

  spectra_dataset = MolnetDataset(mfp, f'{dataset_name}')
  hamming_spectra = MolnetHammingSpectra(spectra_dataset, binary=False)
  hamming_spectra.pre_calculate_spectra_properties(f'{dataset_name}', force_recalculate=False)
  hamming_spectra.generate_spectra_splits(**spectra_parameters)

  stats = hamming_spectra.return_all_split_stats()
  stats_df = pd.DataFrame(stats).sort_values(by='SPECTRA_parameter', ascending=True)

  return stats_df

if __name__ == "__main__":
    datasets = ['tox21']
    #datasets = ['bace_classification', 'bbbp', 'sider', 'clintox', 'delaney', 'freesolv', 'lipo']
    spectra_parameters = {'number_repeats': 3,
                          'random_seed': [42, 44, 46],
                          'spectral_parameters': ["{:.2f}".format(i) for i in np.arange(0, 1.05, 0.05)],
                          'force_reconstruct': False,
                          }

    for dataset in datasets:
        generate_spectra_tanimoto_splits(dataset, spectra_parameters)
        generate_spectra_hamming_splits(dataset, spectra_parameters)
