import json, h5py, matplotlib
import matplotlib.pyplot as plt
matplotlib.use('WebAgg')


data_path = 'D:/OneDrive - sjtu.edu.cn/MyProjects/2023_SSL/data/data_processed/amass_small.h5'


with h5py.File(data_path, 'r') as hf:
    data_columns = json.loads(hf.attrs['columns'])
    dset_data = {dset_: data_[:] for dset_, data_ in hf.items()}

for dset_, data_ in dset_data.items():
    plt.figure()
    plt.title(dset_)
    for i_sample in range(data_.shape[0]):
        plt.plot(data_[i_sample, :, :].ravel(), 'C0')
plt.show()
plt.close('all')








