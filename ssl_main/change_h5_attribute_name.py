import h5py
import json

# to_dump = {}
# data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/sun_drop_jump_backup.h5'
# with h5py.File(data_path, 'r+') as hf:
#     for sub_ in hf.keys():
#         data_columns = list(hf[sub_].attrs['columns'])
#         data_columns = [col.replace('CHEST', 'TRUNK') for col in data_columns]
#         data_columns = [col.replace('WAIST', 'PELVIS') for col in data_columns]
#         to_dump[sub_] = data_columns
#         x=1
#
# data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/sun_drop_jump.h5'
# with h5py.File(data_path, 'r+') as hf:
#     for sub_ in hf.keys():
#         hf[sub_].attrs['columns'] = json.dumps(to_dump[sub_])
#         data_columns = json.loads(hf.attrs['columns'])


data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/hw_running.h5'
with h5py.File(data_path, 'r+') as hf:
    data_columns = list(hf.attrs['columns'][1:-1].split('", "'))
    data_columns[0] = data_columns[0][1:]
    data_columns[-1] = data_columns[-1][:-1]
    data_columns = [col.replace('trunk', 'TRUNK') for col in data_columns]
    data_columns = [col.replace('pelvis', 'PELVIS') for col in data_columns]
    data_columns = [col.replace('thigh', 'THIGH') for col in data_columns]
    data_columns = [col.replace('shank', 'SHANK') for col in data_columns]
    data_columns = [col.replace('foot', 'FOOT') for col in data_columns]
    data_columns = [col.replace('l_', 'L_') for col in data_columns]
    data_columns = [col.replace('r_', 'R_') for col in data_columns]
    data_columns = [col.replace('eL_', 'el_') for col in data_columns]
    to_dump = data_columns

data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/hw_running.h5'
with h5py.File(data_path, 'r+') as hf:
    hf.attrs['columns'] = json.dumps(to_dump)
    data_columns = json.loads(hf.attrs['columns'])
