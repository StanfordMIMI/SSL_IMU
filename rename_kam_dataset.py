import h5py
from const import DATA_PATH, IMU_SEGMENT_LIST
import json


with h5py.File(DATA_PATH + 'all_17_subjects.h5', 'r+') as hf:
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
            column_new = 'kam'
            data_columns[i] = column_new
        if 'EXT_KM_Y' == column:
            column_new = 'kam'
            data_columns[i] = column_new
    hf.attrs['columns'] = json.dumps(data_columns)
