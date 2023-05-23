import torch
import numpy as np
from torch.utils.data import Dataset

class evaluate_dataset(Dataset):
  def __init__(self, data, inE, label, device = 'cpu'):

    self.data  = data
    self.inE   = inE
    self.label = label
    

class evaluator:

  def __init__(self, base_dataset_name, gen_dataset_name, device):
  '''
  base_dataset: Geant4 dataset file name, should be in pt format.
  gen_dataset:  Generative dataset file name, should also be in pt format.
  '''  
    self.base_dataset = torch.load(base_dataset_name, map_location = torch.device(device))
    self.gen_dataset  = torch.load(gen_dataset_name,  map_location = torch.device(device))
    self.dataset_size = min(len(self.base_dataset),len(self.gen_dataset_name))
    self.base_data    = self.base_dataset[0][:self.dataset_size]
    self.gen_data     = self.gen_dataset[0][:self.dataset_size]
    self.base_inE     = self.base_dataset[1][:self.dataset_size]
    self.gen_inE      = self.gen_dataset[1][:self.dataset_size]
    self.base_label   = tensor.ones(self.dataset_size, device=device)
    self.gen_label    = tensor.zeros(self.dataset_size, device=device)

  def separate_ttv(train_ratio, test_ratio):
    assert (train_ratio + test_ratio) <= 1.0
    train_size = int(2 * self.dataset_size * train_ratio)
    test_size  = int(2 * self.dataset_size * test_ratio)
  
