import h5py
import json

to_dump = {}
data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/sun_drop_jump_backup.h5'
with h5py.File(data_path, 'r+') as hf:
    for sub_ in hf.keys():
        data_columns = list(hf[sub_].attrs['columns'])
        data_columns = [col.replace('CHEST', 'TRUNK') for col in data_columns]
        data_columns = [col.replace('WAIST', 'PELVIS') for col in data_columns]
        to_dump[sub_] = data_columns
        x=1

data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/sun_drop_jump.h5'
with h5py.File(data_path, 'r+') as hf:
    for sub_ in hf.keys():
        hf[sub_].attrs['columns'] = json.dumps(to_dump[sub_])
        data_columns = json.loads(hf.attrs['columns'])
        x=1
