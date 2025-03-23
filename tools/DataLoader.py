import random


class DataLoader:
    def __init__(self, len, batch_size, shuffle=True):
        """
        初始化 DataLoader
        :param len: 数据集（列表、NumPy 数组等）大小
        :param batch_size: 批次大小
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_samples = len
        self.data = list(range(0, self.num_samples))
        self.num_batches = max((self.num_samples // self.batch_size), 1)

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        start = 0
        while start < self.num_samples:
            end = start + self.batch_size
            # 如果最后一组不足 batch_size，则合并到前一组
            if end + self.batch_size > self.num_samples:
                end = self.num_samples
            batch = self.data[start:end]
            yield batch
            start = end

    def __len__(self):
        """
        返回批次数量
        """
        return self.num_batches


# 示例使用
if __name__ == "__main__":
    # 示例数据
    data = list(range(100))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 11

    # 创建 DataLoader
    dataloader = DataLoader(data.__len__(), batch_size)

    # 遍历 DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}: {batch}")
