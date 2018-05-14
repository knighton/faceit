import numpy as np


class Split(object):
    def __init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                 y_sample_shapes, y_dtypes):
        self.samples_per_epoch = samples_per_epoch

        self.sample_shapes = x_sample_shapes, y_sample_shapes
        self.x_sample_shapes = x_sample_shapes
        self.y_sample_shapes = y_sample_shapes

        self.dtypes = x_dtypes, y_dtypes
        self.x_dtypes = x_dtypes
        self.y_dtypes = y_dtypes

    def batches_per_epoch(self, batch_size):
        return self.samples_per_epoch // batch_size

    def x_batch_shapes(self, batch_size):
        return [(batch_size,) + x for x in self.x_sample_shapes]

    def y_batch_shapes(self, batch_size):
        return [(batch_size,) + y for y in self.y_sample_shapes]

    def batch_shapes(self, batch_size):
        x = self.x_batch_shapes(batch_size),
        y = self.y_batch_shapes(batch_size)
        return x, y

    def get_batch(self, batch_size, index):
        raise NotImplementedError

    def shuffle(self, batch_size):
        batches_per_epoch = self.batches_per_epoch(batch_size)
        x = np.arange(batches_per_epoch)
        np.random.shuffle(x)
        return x


class RamSplit(Split):
    @classmethod
    def normalize(cls, xx):
        if isinstance(xx, np.ndarray):
            xx = [xx]
        else:
            assert isinstance(xx, (list, tuple))
        return xx

    @classmethod
    def check(cls, xx, yy):
        counts = set()
        for x in xx:
            assert isinstance(x, np.ndarray)
            counts.add(len(x))
        for y in yy:
            assert isinstance(y, np.ndarray)
            counts.add(len(y))
        assert len(counts) == 1
        assert counts.pop()

    def __init__(self, xx, yy):
        xx = self.normalize(xx)
        yy = self.normalize(yy)
        self.check(xx, yy)
        samples_per_epoch = len(xx[0])
        x_sample_shapes = [x[0].shape for x in xx]
        x_dtypes = [x[0].dtype.name for x in xx]
        y_sample_shapes = [y[0].shape for y in yy]
        y_dtypes = [y[0].dtype.name for y in yy]
        Split.__init__(self, samples_per_epoch, x_sample_shapes, x_dtypes,
                       y_sample_shapes, y_dtypes)
        self.xx = xx
        self.yy = yy

    def get_batch(self, batch_size, index):
        a = index * batch_size
        z = (index + 1) * batch_size
        batch_xx = [x[a:z] for x in self.xx]
        batch_yy = [y[a:z] for y in self.yy]
        return batch_xx, batch_yy


class Dataset(object):
    @classmethod
    def split(cls, xx, yy, val_frac):
        xx = RamSplit.normalize(xx)
        yy = RamSplit.normalize(yy)
        n = len(xx[0])
        for x in xx[1:]:
            assert len(x) == n
        for y in yy:
            assert len(y) == n
        x = tuple(zip(*xx))
        y = tuple(zip(*yy))
        pairs = list(zip(x, y))
        np.random.shuffle(pairs)
        index = int(len(pairs) * val_frac)

        x_train, y_train = zip(*pairs[index:])
        x_train = list(zip(*x_train))
        y_train = list(zip(*y_train))
        for i, z in enumerate(x_train):
            x_train[i] = np.array(z, z[0].dtype)
        for i, z in enumerate(y_train):
            y_train[i] = np.array(z, z[0].dtype)
        train = RamSplit(x_train, y_train)

        x_val, y_val = zip(*pairs[:index])
        x_val = list(zip(*x_val))
        y_val = list(zip(*y_val))
        for i, z in enumerate(x_val):
            x_val[i] = np.array(z, z[0].dtype)
        for i, z in enumerate(y_val):
            y_val[i] = np.array(z, z[0].dtype)
        val = RamSplit(x_val, y_val)

        return cls(train, val)

    def __init__(self, train, val):
        assert isinstance(train, Split)
        if val is not None:
            assert isinstance(val, Split)
            assert train.sample_shapes == val.sample_shapes
            assert train.dtypes == val.dtypes
        self.train = train
        self.val = val

        if val:
            self.samples_per_epoch = \
                train.samples_per_epoch + val.samples_per_epoch
        else:
            self.samples_per_epoch = train.samples_per_epoch

        self.sample_shapes = train.sample_shapes
        self.x_sample_shapes = train.x_sample_shapes
        self.y_sample_shapes = train.y_sample_shapes

        self.dtypes = train.dtypes
        self.x_dtypes = train.x_dtypes
        self.y_dtypes = train.y_dtypes

    def batches_per_epoch(self, batch_size, max_batches_per_epoch=None):
        n = self.train.batches_per_epoch(batch_size)

        if self.val:
            n += self.val.batches_per_epoch(batch_size)

        z = max_batches_per_epoch
        if isinstance(z, int) and 0 <= z and z < n:
            n = z

        return n

    def get_batch(self, batch_size, is_training, index):
        if is_training:
            split = self.train
        else:
            split = self.val
        return split.get_batch(batch_size, index)

    def shuffle(self, batch_size):
        num_train_batches = self.train.batches_per_epoch(batch_size)
        if self.val:
            num_val_batches = self.val.batches_per_epoch(batch_size)
        else:
            num_val_batches = 0
        train_batches = np.arange(num_train_batches)
        val_batches = np.arange(num_val_batches)
        x = np.zeros((num_train_batches + num_val_batches, 2), 'int64')
        x[train_batches, 0] = 1
        x[train_batches, 1] = train_batches
        x[num_train_batches + val_batches, 1] = val_batches
        np.random.shuffle(x)
        return x

    def each_batch(self, batch_size, max_batches_per_epoch=None):
        i = 0
        for is_training, index in self.shuffle(batch_size):
            if i == max_batches_per_epoch:
                return
            xx, yy = self.get_batch(batch_size, is_training, index)
            yield is_training, xx, yy
            i += 1
