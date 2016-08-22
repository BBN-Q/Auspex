import numpy as np
import pandas as pd
import itertools
import h5py

samps        = np.linspace(0,1023,1024)
reps         = np.linspace(50,55,3)
phase_points = np.linspace(0,10,300)
fields       = np.linspace(10,20,3)

params = {'test': 1, 'test2': "AAA", 'test3': 1.24}

import time
a =      list(itertools.product([fields[0]], np.linspace(0,12,4), reps, samps ))
a.extend(list(itertools.product([fields[1]], np.linspace(0,12,4), reps, samps )))
a.extend(list(itertools.product([fields[2]], np.linspace(0,5,2),  reps, samps )))
a.extend(list(itertools.product([fields[0]], np.linspace(0,12,4), reps, samps )))
a.extend(list(itertools.product([fields[1]], np.linspace(0,12,4), reps, samps )))
a.extend(list(itertools.product([fields[2]], np.linspace(0,5,2),  reps, samps )))
a.extend(list(itertools.product([fields[0]], np.linspace(0,12,4), reps, samps )))
a.extend(list(itertools.product([fields[1]], np.linspace(0,12,4), reps, samps )))
a.extend(list(itertools.product([fields[2]], np.linspace(0,5,2),  reps, samps )))
# a.append(itertools.product(samps, reps, phase_points, 10.0))

# print("Length of a is", len(a))
# print("Expected #bytes", 8*len(a))

index = pd.MultiIndex.from_tuples(a, names=['Fields', 'Phase_Points', 'Reps' ,'Samples'])

s = pd.DataFrame(np.random.randn(len(a)), index=index)

store = pd.HDFStore('dat_frame_test.h5')
store_compressed = pd.HDFStore('dat_frame_test_compressed.h5', complevel=9, complib='blosc')

start = time.time()
store['data'] = s
print("Base: ", time.time()-start)

# store.get_storer('data').attrs.metadata = params
for k, v in params.items():
	store.get_storer('data').attrs[k] = v
start = time.time()
store_compressed['data'] = s
print("Compressed: ", time.time()-start)
# store_compressed['params'] = pdf
# print(s)


a =  list(itertools.product([fields[0]], np.linspace(0,12,4), reps, samps ))
index = pd.MultiIndex.from_tuples(a, names=['Fields', 'Phase_Points', 'Reps' ,'Samples'])
s = pd.DataFrame(np.random.randn(len(a)), index=index)

a1 =  list(itertools.product([fields[1]], np.linspace(0,12,4), reps, samps ))
index1 = pd.MultiIndex.from_tuples(a1, names=['Fields', 'Phase_Points', 'Reps' ,'Samples'])
s1 = pd.DataFrame(np.random.randn(len(a1)), index=index1)

a2 =  list(itertools.product([fields[2]], np.linspace(0,5,2), reps, samps ))
index2 = pd.MultiIndex.from_tuples(a2, names=['Fields', 'Phase_Points', 'Reps' ,'Samples'])
s2 = pd.DataFrame(np.random.randn(len(a2)), index=index2)

store_append = pd.HDFStore('dat_frame_test_append.h5')

start = time.time()
store_append.append('data', s, index=False)
store_append.append('data', s1, index=False)
store_append.append('data', s2, index=False)
store_append.append('data', s, index=False)
store_append.append('data', s1, index=False)
store_append.append('data', s2, index=False)
store_append.append('data', s, index=False)
store_append.append('data', s1, index=False)
store_append.append('data', s2, index=False)
print("Append no index: ", time.time()-start)

# store_append.create_table_index('data', columns=list(s.index.names))

store_append_compressed = pd.HDFStore('dat_frame_test_append_compressed.h5', complevel=3, complib='blosc')

start = time.time()
store_append_compressed.append('data', s, index=False)
store_append_compressed.append('data', s1, index=False)
store_append_compressed.append('data', s2, index=False)
store_append_compressed.append('data', s, index=False)
store_append_compressed.append('data', s1, index=False)
store_append_compressed.append('data', s2, index=False)
store_append_compressed.append('data', s, index=False)
store_append_compressed.append('data', s1, index=False)
store_append_compressed.append('data', s2, index=False)
print("Append compressed no index: ", time.time()-start)

# store_append.create_table_index('data', columns=list(s.index.names))

store_append_index = pd.HDFStore('dat_frame_test_append_index.h5')

start = time.time()
store_append_index.append('data', s, index=False)
store_append_index.append('data', s1, index=False)
store_append_index.append('data', s2, index=False)
store_append_index.create_table_index('data', columns=list(s.index.names))
print("Append with index: ", time.time()-start)


store_append_compressed_index = pd.HDFStore('dat_frame_test_append_compressed_index.h5', complevel=3, complib='blosc')

start = time.time()
store_append_compressed_index.append('data', s, index=False)
store_append_compressed_index.append('data', s1, index=False)
store_append_compressed_index.append('data', s2, index=False)
store_append_compressed_index.create_table_index('data', columns=list(s.index.names))
print("Append compressed with index: ", time.time()-start)


# Now read in a raw HDF5 array as a pandas array:
samps        = np.linspace(0,1023,1024)
reps         = np.linspace(50,55,3)
phase_points = np.linspace(0,10,300)
fields       = np.linspace(10,20,3)

all_tuples = np.array(list(itertools.product(fields, phase_points, reps, samps )))
print(all_tuples)
data       = np.random.random(len(all_tuples))

f = h5py.File('straight-h5.h5', 'w')
df = f.create_dataset('data', (len(all_tuples), 4+1), dtype='f', chunks=True, maxshape=(None, 4+1))
df[:,-1] = data
df[:,:-1] = all_tuples
f.close()

f = h5py.File('straight-h5-lzf.h5', 'w')
df = f.create_dataset('data', (len(all_tuples), 4+1), dtype='f', compression='lzf', chunks=True, maxshape=(None, 4+1))
df[:,-1] = data
df[:,:-1] = all_tuples
f.close()

f = h5py.File('straight-h5-gzip.h5', 'w')
df = f.create_dataset('data', (len(all_tuples), 4+1), dtype='f', compression='gzip', chunks=True, maxshape=(None, 4+1))
df[:,-1] = data
df[:,:-1] = all_tuples
f.close()

with h5py.File('straight-h5.h5', 'r') as f:
	dat = f['data'][:]
	axis_values = [np.array(v) for v in dat[:,:-1].T] 
	s = pd.DataFrame(dat[:,-1], index=axis_values)
	s.to_hdf('rewrite.h5', 'data', complevel=3, complib='blosc', index=False)
	print(s)