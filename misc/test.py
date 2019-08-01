import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_EPOCHS = 10
N_EXECUTIONS = 5

epochs = range(1, N_EPOCHS + 1)
executions = range(1, N_EXECUTIONS + 1)

loss = np.random.randn(N_EPOCHS, N_EXECUTIONS)
acc = np.random.randn(N_EPOCHS, N_EXECUTIONS)

loss_array = xr.DataArray(loss, dims=('epoch', 'execution'), coords={'epoch': range(1, N_EPOCHS + 1),
                                                                     'execution': range(1, N_EXECUTIONS + 1)})

acc_array = xr.DataArray(acc, dims=('epoch', 'execution'), coords={'epoch': range(1, N_EPOCHS + 1),
                                                                   'execution': range(1, N_EXECUTIONS + 1)})

dataset1 = xr.Dataset({'loss': loss_array, 'acc': acc_array})

dataset2 = xr.Dataset({'loss': (['epoch', 'execution'], loss),
                       'acc': (['epoch', 'execution'], acc)},
                      coords={'epoch': range(1, N_EPOCHS + 1),
                              'execution': range(1, N_EXECUTIONS + 1)})
                              # 'dataset': ['MNIST', 'cifar10', 'cifar100']})


b = dataset2.to_array()
print(b.to_dataframe('value'))

# print(loss_array.loc[1,:])
# print(loss_array)
# dataset1.mean(dim='epoch').to_dataframe().plot()
# plt.plot()

# print(dataset1)
#
# temp = 15 + 8 * np.random.randn(2, 2, 3)
# precip = 10 * np.random.rand(2, 2, 3)
# lon = [[-99.83, -99.32], [-99.79, -99.23]]
# lat = [[42.25, 42.21], [42.63, 42.59]]
#
# ds = xr.Dataset()
# ds['temperature'] = (('x', 'y', 'time'), temp)
# ds['precipitation'] = (('x', 'y', 'time'), precip)
# ds.coords['lat'] = (('x', 'y'), lat)
# ds.coords['lon'] = (('x', 'y'), lon)
# ds.coords['time'] = pd.date_range('2014-09-06', periods=3)
# ds.coords['reference_time'] = pd.Timestamp('2014-09-05')
#
# print(ds)