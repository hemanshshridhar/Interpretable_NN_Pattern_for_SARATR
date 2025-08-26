import os
import numpy as np
pols = ['HH', 'HV', 'VH', 'VV']
data_dir = '/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA'
target_coordinate_centers = {'chevimalibu':np.array([20.66,-18.71,0.02]),'Toyotacamary':np.array([9.97,-5.22,0.02]), 'Fordtaurus':np.array([12.43,-18.21,-0.05]),'casetractor':np.array([-0.96,-17.48,-0.17]),'hysterforklift':np.array([24.96,-6.45,0.20]),'nissanmaximal':np.array([31.42,-28.87,-0.07]),'nissancentra':np.array([22.68,-28.30,-0.17])}
for target in target_coordinate_centers:
  for p in range(7):
    for pol in pols:
      for az in range(356):
        az_str = f'{az:03d}'  # Zero-padded azimuth

        filename = f'data_3dsar_pass{p}_az{az_str}_{pol}.mat'
        path = os.path.join(data_dir, pol, filename)
        pass_prefix =
        data_path = os.path.join(data_dir,pass_prefix)
        [phs,platform] = AFRL(data_dir, pol, az +1,n_az =4)
        img_plane = img_plane_dict(platform, res_factor = 1.4, upsample = True, aspect = 1.0)
        img = DSBP(phs, platform, img_plane, center= target_coordinate_centers[target] , size = [54,54] , derate = 1.05, taylor = 47, n = 32, beta = 4, cutoff = 'nyq', factor_max = 10, factor_min = 0)
        path = f'data_3dsar_pass8_az{az_str}_{pol}.mat'
        save_dir = '/content/drive/MyDrive/Polsar/GOTCHA-CP_Disc2/DATA/pass8'
        pas = f'pass{p}'
        prefix = f'{pol}_NPY'
        label  = f'{target}'
        save_dir = os.path.join(data_dir,label,pas,prefix)
        os.makedirs(save_dir, exist_ok=True)
        file_path =  f'data_3dsar_pass{p}_az{az_str}_{pol}.npy'
        save_path = os.path.join(save_dir, file_path)
        np.save(save_path, img)

        print(f"Image saved to: {save_path}")


