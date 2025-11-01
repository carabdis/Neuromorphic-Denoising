import torch

batch_size = 1
in_channel = 3
out_channel = 4
init = torch.zeros((batch_size, in_channel))
weight = torch.zeros((out_channel, in_channel))
init[0, 0] = 1
weight[0, 0] = 1

print(torch.nn.functional.linear(init, weight))

for i in range(out_channel):
    result = 0
    for j in range(in_channel):
        result += init[0, j] * weight[i, j]
    print(result, end=' ')