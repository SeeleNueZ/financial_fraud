import random


class DataLoader:
    def __init__(self, len, batch_size,shuffle=True):
        """
        初始化 DataLoader
        :param data: 数据集（列表、NumPy 数组等）
        :param batch_size: 批次大小
        """
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_samples = len
        self.data = list(range(0, self.num_samples))
        self.num_batches = self.num_samples // self.batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.data)
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = start + self.batch_size
            batch = self.data[start:end]
            yield batch

    def __len__(self):
        """
        返回批次数量
        """
        return self.num_batches


# 示例使用
if __name__ == "__main__":
    # 示例数据
    data = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    batch_size = 3

    # 创建 DataLoader
    dataloader = DataLoader(data.__len__(), batch_size)

    # 遍历 DataLoader
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}: {batch}")
