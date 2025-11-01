import numpy as np
import torch


x = torch.tensor(np.random.gamma(1, 1, torch.tensor(np.random.random((4, 5))).shape))
print(x)
