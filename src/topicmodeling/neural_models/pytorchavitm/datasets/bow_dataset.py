# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from scipy import sparse

class BOWDataset(Dataset):
    
    """Class to load BOW dataset."""

    def __init__(self, X, idx2token, cv):

        """
        Initializes BOWDataset.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Document-term matrix
        idx2token : list
            A list of feature names
        """

        self.X = X
        self.idx2token = idx2token
        self.cv = cv

    def __len__(self):
        """Returns length of dataset."""
        return self.X.shape[0]

    def __getitem__(self, i):
        """Returns sample from dataset at index i."""
        if type(self.X[i]) == sparse.csr_matrix:
            X = torch.FloatTensor(self.X[i].todense())
        else:
            X = torch.FloatTensor(self.X[i])
        
        return {'X': X}
