from torch.utils.data import Dataset
import time
import numpy as np
import torch


class DataLoader():
    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=4):
        self.data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.data_loader_iter = self.data_loader._get_iterator()

    def build(self):
        # self.loader = iter(loader)
        self.data_loader_iter = self.data_loader._get_iterator()

    def next(self):
        try:
            # batch = next(self.loader)
            batch = self.data_loader_iter.__next__()
        except StopIteration:
            self.build()
            return None
        return batch


class DataLoaderDived():
    def __init__(self, dataset, base_batch_size=4, shuffle=True, num_workers=4, batch_size=1):
        assert base_batch_size % batch_size == 0
        self.divide_length = base_batch_size // batch_size
        self.base_batch_size = base_batch_size
        self.batch_size = batch_size
        self.base_loader = DataLoader(dataset=dataset, batch_size=base_batch_size, shuffle=shuffle, num_workers=num_workers)
        self.cnt = 0
        self.tmp_batch = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.next()
        if batch is None:
            raise StopIteration
        else:
            return batch

    def next(self):
        if self.tmp_batch is None:
            # print('get new base batch')
            batch_data = self.base_loader.next()
            if batch_data is None:  # end of epoch
                return None
            self.tmp_batch = batch_data
        sub_batch = self.get_sub_batch()
        return sub_batch

    def get_sub_batch(self):
        start = self.cnt
        end = self.cnt + self.batch_size

        if isinstance(self.tmp_batch, list) or isinstance(self.tmp_batch, tuple):
            new_batch = []
            for data in self.tmp_batch:
                new_batch.append(data[start:end])
        elif isinstance(self.tmp_batch, dict):
            new_batch = {}
            for kk in self.tmp_batch.keys():
                new_batch[kk] = self.tmp_batch[kk][start:end]
        else:  # just one tensor
            new_batch = self.tmp_batch[start:end]
        self.cnt = end
        if end == self.base_batch_size:
            self.cnt = 0
            self.tmp_batch = None
        return new_batch

    @classmethod
    def demo(cls):

        class DemoData(Dataset):
            def __init__(self):
                self.data = list(range(1000))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                time.sleep(0.1)
                res = np.zeros((3, 100, 100))
                res = torch.from_numpy(res).float()
                return {0: res, 1: res}

        data_loader = DataLoaderDived(DemoData(), base_batch_size=8, batch_size=2, shuffle=True, num_workers=8)
        # for cnt in range(100):
        #     st = time.time()
        #     batch = data_loader.next()
        #     end = time.time()
        #     if batch is None:
        #         continue
        #     print(f"{cnt} use time: {end - st:.2f}, {batch[0].shape}")
        for cnt, batch in enumerate(data_loader):
            if batch is None:
                continue
            print(f"{cnt}  {batch[0].shape}")


if __name__ == '__main__':
    DataLoaderDived.demo()
