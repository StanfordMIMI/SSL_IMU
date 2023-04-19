import h5py
import json
import numpy as np
from config import DATA_PATH


def rename_kam_dataset():
    with h5py.File(DATA_PATH + 'walking_knee_moment.h5', 'r+') as hf:
        data_columns = json.loads(hf.attrs['columns'])
        for i, column in enumerate(data_columns):
            if 'Accel' == column[:5]:
                axis, segment = column.split('Accel')[1][0], column.split('Accel')[1][2:]
                column_new = segment + '_Accel_' + axis
                data_columns[i] = column_new
            if 'Gyro' == column[:4]:
                axis, segment = column.split('Gyro')[1][0], column.split('Gyro')[1][2:]
                column_new = segment + '_Gyro_' + axis
                data_columns[i] = column_new
            if 'EXT_KM_X' == column:
                column_new = 'KFM'
                data_columns[i] = column_new
            if 'EXT_KM_Y' == column:
                column_new = 'KAM'
                data_columns[i] = column_new
        if 'sub_id' not in data_columns:
            for sub_name, data in hf.items():
                data_ = data[:]
                subject_id_array = np.full([data_.shape[0], data_.shape[1], 1], int(sub_name[-2:])-1)
                data_ = np.concatenate([subject_id_array, data_], axis=2)
                del hf[sub_name]
                hf.create_dataset(sub_name, data_.shape, data=data_)
            data_columns.insert(0, 'sub_id')
        hf.attrs['columns'] = json.dumps(data_columns)


if __name__ == '__main__':
    rename_kam_dataset()


