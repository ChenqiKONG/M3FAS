import pandas as pd
from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset

class face_datareader(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_path_details = pd.read_csv(csv_file,header=None)
        self.transform = transform

    def __len__(self):
        return len(self.data_path_details)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        face_path = self.data_path_details.iloc[idx, 0]
        #face_path = face_path.replace('/EchoFAS_dataset/', '/EchoFAS_dataset_M3FAS/')
        face = io.imread(face_path)
        spect_path = self.data_path_details.iloc[idx, 1]
        #spect_path = spect_path.replace('/EchoFAS_dataset/', '/EchoFAS_dataset_M3FAS/')
        spect = np.load(spect_path)
        label = self.data_path_details.iloc[idx, 2]
        samples = {'faces': face,
                  'spects': spect,
                  'labels': label}
        if self.transform:
            samples = self.transform(samples)
        return samples

class my_transforms(object):
    def __init__(self, size=128, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        self.size = (size, size)

    def __call__(self, sample):
        faces_ori = sample['faces']
        spects = sample['spects']
        labels = sample['labels']
        faces = transform.resize(faces_ori, self.size)
        faces = faces.transpose(2, 0, 1)
        spects = spects.transpose(2, 0, 1)
        ### to pytorch tensor
        faces = torch.from_numpy(faces.copy()).float() 
        spects = torch.from_numpy(spects.copy()).float() 
        labels = torch.tensor(labels).long()
        ### normalize
        faces = (faces - self.mean) / self.std
        ### output
        sample_tran = {'faces': faces,
                  'spects': spects,
                  'labels': labels}
        return sample_tran