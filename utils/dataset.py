import os
import numpy as np
from torch.utils.data import Dataset
from .datapreprocess import complex_correlation, rotate

class PolarimetricSARDataset(Dataset):
    def __init__(self, root_dir, channel_dirs, rotate =None, correlation=None, device ='cuda'):
        self.root_dir = root_dir
        self.channel_dirs = channel_dirs
        self.rotate= rotate
        self.correlation = correlation
        self.data = self._collect_npy_file_tuples()
        self.label_to_index = {label: idx for idx, label in enumerate(sorted(set(lbl for *_, lbl in self.data)))}
        self.device  = device
    def _collect_npy_file_tuples(self):
        data = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            full_channel_dirs = [os.path.join(class_dir, ch) for ch in self.channel_dirs]
            if not all(os.path.isdir(ch_dir) for ch_dir in full_channel_dirs):
                continue
            # print(full_channel_dirs)
            file_names = [sorted([f for f in os.listdir(d) if f.endswith('.npy')]) for d in full_channel_dirs]
            for i in range(len(file_names[0])):
                data.append((
                    os.path.join(full_channel_dirs[0], file_names[0][i]),
                    os.path.join(full_channel_dirs[1], file_names[1][i]),
                    os.path.join(full_channel_dirs[2], file_names[2][i]),
                    os.path.join(full_channel_dirs[3], file_names[3][i]),
                    class_name
                ))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f_HH, f_HV, f_VH, f_VV, label = self.data[idx]

        S_HH = np.load(f_HH)
        S_HV = np.load(f_HV)
        S_VH = np.load(f_VH)
        S_VV = np.load(f_VV)
        S_HH = torch.tensor(S_HH)
        S_HV = torch.tensor(S_HV)
        S_VH = torch.tensor(S_VH)
        S_VV = torch.tensor(S_VV)
        S_VV = S_VV.to(self.device)
        S_VH = S_VH.to(self.device)
        S_HV = S_HV.to(self.device)
        S_HH = S_HH.to(self.device)
        C1, C2, C3, C4 = [], [], [], []
         
        for i in range(0, 360, 5):
            SHV_rot, SVV_rot, SHH_rot, SHH_plus_SVV, SHH_minus_SVV = self.rotate(S_HH, S_HV, S_VH, S_VV, i)
            # SHV_rot = torch.tensor(SHV_rot).to(device)
            # SVV_rot = torch.tensor(SVV_rot).to(device)
            # SHH_rot = torch.tensor(SHH_rot).to(device)
            # SHH_plus_SVV =  torch.tensor(SHH_plus_SVV).to(device)
            # SHH_minus_SVV = torch.tensor(SHH_minus_SVV).to(device)
            
            # SHV_rot = SHV_rot.to(device)
            # SVV_rot = SVV_rot.to(device)
            # SHH_rot = SHH_rot.to(device)
            # SHH_plus_SVV =  SHH_plus_SVV.to(device)
            # SHH_minus_SVV = SHH_minus_SVV.to(device)            
            P1 = complex_correlation(SHH_rot, SHV_rot)
            P2 = complex_correlation(SHH_rot, SVV_rot)
            P3 = complex_correlation(SHH_plus_SVV, SHH_minus_SVV)
            P4 = complex_correlation(SHH_minus_SVV, SHV_rot)

            C1.append(P1)
            C2.append(P2)
            C3.append(P3)
            C4.append(P4)

        # Shape: (72, h, w)
        C1_array = torch.stack(C1, dim=0)
        C1_array = C1_array.unsqueeze(0)
        C2_array = torch.stack(C2, dim=0)
        C2_array = C2_array.unsqueeze(0)
        C3_array = torch.stack(C3, dim=0)
        C3_array = C3_array.unsqueeze(0)
        C4_array = torch.stack(C4, dim=0)
        C4_array = C4_array.unsqueeze(0)
    
        # Optionally, you can stack these into a (4, 72, h, w) tensor
        channels = [C1_array, C2_array, C3_array, C4_array]
        stacked = torch.stack(channels, dim=0)
        
        stacked = stacked.squeeze()
        label_idx = self.label_to_index[label]
        return stacked, label_idx