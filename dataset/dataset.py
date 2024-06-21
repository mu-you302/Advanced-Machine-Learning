import random
import numpy as np
from torch.utils.data.dataset import Dataset
from config import cfg


# 组合多个数据集，可以选择使它们具有相同的长度或保持其原始长度
class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # 所有数据库具有相同的长度
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # 每个数据库具有不同的长度
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = index // self.max_db_data_num
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (
                self.max_db_data_num // len(self.dbs[db_idx])
            ):  # 最后一个批次随机采样
                data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
            else:  # 在最后一个批次之前使用模数
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx - 1]

        return self.dbs[db_idx][data_idx]
