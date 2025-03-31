import torch.nn as nn


class LR1_g(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(LR1_g, self).__init__()
        layers = []
        prev_size = input_size

        # 添加中间隐藏层
        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size  # 下一层的输入是当前层的输出

        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))

        # 将所有层组合成序列
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 示例用法
if __name__ == "__main__":
    # 定义模型参数：输入1，中间层[1,2]，输出3
    model = LR1_g(input_size=6, hidden_sizes=[8, 32, 8], output_size=6)

    # 打印模型结构
    print(model)

    # 输出模型各层详情
    # from torchsummary import summary

    # summary(model, (1,))  # 输入大小为1
