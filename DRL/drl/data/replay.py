import numpy as np
from drl.data import Batch

class Storage:

    def __init__(self, size):
        self._maxsize = size
        self._size = 0
        self._index = 0
        self._keys = []
        self.reset()


    def __len__(self):
        return self._size


    def __del__(self):
        for k in self._keys:
            v = getattr(self, k)
            del v

    def __getitem__(self, index):
        batch_dict = dict(
            zip(self._keys, [getattr(self,k)[index] for k in self._keys]))
        return batch_dict


    def set_placeholder(self, key, value):
        if isinstance(value, np.ndarray):
            setattr(self, key, np.zeros((self._maxsize, *value.shape)))
        elif isinstance(value, dict):
            setattr(self, key, np.array([{} for _ in range(self._maxsize)]))
        elif np.isscalar(value):
            setattr(self, key, np.zeros((self._maxsize,)))

    
    def add(self, data):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v)
            getattr(self, k)[self._index] = v

        self._size = min(self._size + 1, self._maxsize)
        self._index = (self._index + 1) % self._maxsize


    def add_list(self, data, length):
        assert isinstance(data, dict)

        _tmp_idx = self._index + length

        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                self.set_placeholder(k, v[0])
            
            assert v.shape[0] == length
            
            if _tmp_idx < self._maxsize:
                getattr(self, k)[self._index:_tmp_idx] = v
            else:
                getattr(self, k)[self._index:] = v[:self._maxsize - self._index]
                getattr(self, k)[:_tmp_idx - self._maxsize] = v[self._maxsize - self._index:]
            

        self._size = min(self._size + length, self._maxsize)
        self._index = (self._index + length) % self._maxsize


    def update(self, storage_new):
        i = begin = storage_new._index % len(storage_new)
        while True:
            self.add(storage_new[i])
            i = (i+1)% len(storage_new)
            if i == begin:
                break

    def reset(self):
        self._index = self._size = 0


    def sample(self, batch_size):
        if batch_size > 0:
            indice = np.random.choice(self._size, batch_size)
        else: # sample all available data when batch_size=0
            indice = np.concatenate([
                np.arange(self._index, self._size),
                np.arange(0, self._index),
            ])
        return Batch(**self[indice]), indice


class CacheBuffer(Storage):

    def __init__(self):
        super().__init__(size=0)

    def add(self, data):
        assert isinstance(data, dict)
        for k, v in data.items():
            if v is None:
                continue
            if k not in self._keys:
                self._keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

        self._index += 1
        self._size += 1

    def reset(self):
        self._index = self._size = 0
        for k in self._keys:
            setattr(self, k, [])