from .stream import DataStreamDescriptor, DataAxis
import numpy as np
import os, os.path
import json

class AuspexDataContainer(object):
    def __init__(self, base_path, mode='a'):
        if '.auspex' in base_path:
            base_path = base_path.replace('.auspex', '')
        self.base_path = os.path.abspath(base_path) + '.auspex'
        self.open_mmaps = []
        self.mode = mode
        self._create()
    def close(self):
        for mm in self.open_mmaps:
            del mm
    def _create(self):
        if self.mode not in ['a', 'w+']:
            assert not os.path.exists(self.base_path), "Existing data container found. Did you want to open instead?"
        os.makedirs(self.base_path, exist_ok=True)
        self.groups = {}
    def new_group(self, groupname):
        assert os.path.exists(self.base_path), "No data container found. This should have happened automatically?"
        if self.mode not in ['a', 'w+']:
            assert not os.path.exists(self.base_path), "Existing data container found. Did you want to open instead?"
        os.makedirs(os.path.join(self.base_path,groupname), exist_ok=True)
        self.groups[groupname] = []
    def new_dataset(self, groupname, datasetname, descriptor):
        self.groups[groupname].append(datasetname)
        self._create_meta(groupname, datasetname, descriptor)
        return self._create_memmap(groupname, datasetname, (np.product(descriptor.dims()),), descriptor.dtype)
    def _create_meta(self, groupname, datasetname, descriptor):
        filename = os.path.join(self.base_path,groupname,datasetname+'_meta.json')
        assert not os.path.exists(filename), "Existing dataset metafile found. Did you want to open instead?"
        meta = {'shape': tuple(descriptor.dims()), 'dtype': np.dtype(descriptor.dtype).str}
        meta['axes'] = {a.name: a.points.tolist() for a in descriptor.axes}
        meta['units'] = {a.name: a.unit for a in descriptor.axes}
        meta['meta_data'] = {}
        for a in descriptor.axes:
            if a.metadata is not None:
                meta['meta_data'][a.name] = a.metadata
            else:
                meta['meta_data'][a.name] = None
        meta['filename'] = os.path.join(self.base_path,groupname,datasetname)
        with open(filename, 'w') as f:
            json.dump(meta, f)
    def _create_memmap(self, groupname, datasetname, shape, dtype, mode='w+'):
        filename = os.path.join(self.base_path,groupname,datasetname+'.dat')
        assert not os.path.exists(filename), "Existing dataset found. Did you want to open instead?"
        mm = np.memmap(filename, dtype=dtype, mode=mode, shape=shape)
        self.open_mmaps.append(mm)
        return mm
    def open_all(self):
        ret = {}
        for groupname in os.listdir(self.base_path):
            ret[groupname] = {}
            for datasetname in os.listdir(os.path.join(self.base_path,groupname)):
                if datasetname[-4:] == '.dat':
                    ret[groupname][datasetname[:-4]] = self.open_dataset(groupname, datasetname[:-4])
        return ret
    def open_dataset(self, groupname, datasetname):
        filename = os.path.join(self.base_path,groupname,datasetname+'_meta.json')
        assert os.path.exists(filename), "Could not find dataset. Is this the correct name?"
        with open(filename, 'r') as f:
            meta = json.load(f)
            
        filename = os.path.join(self.base_path,groupname,datasetname+'.dat')
        assert os.path.exists(filename), "Could not find dataset. Is this the correct name?"
        flat_shape = (np.product(meta['shape']),)
        mm = np.memmap(filename, dtype=meta['dtype'], mode='r', shape=flat_shape)
        data = np.array(mm).reshape(tuple(meta['shape']))
        del mm
        
        desc = DataStreamDescriptor(meta['dtype'])
        for name, points in meta['axes'].items():
            ax = DataAxis(name, points, unit=meta['units'][name])
            ax.metadata = meta['meta_data'][name]
            desc.add_axis(ax)
        return data, desc
