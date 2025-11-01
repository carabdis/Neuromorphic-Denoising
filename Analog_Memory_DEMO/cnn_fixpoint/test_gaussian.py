import torch
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def gaussian_noise(inputs, mean, std):  ## generate gaussian noise
    noise = torch.randn(inputs.size())*std + mean
    return noise

def invert_gaussian(inputs, mean, std):  ## generate gaussian noise
    invert = 1/inputs
    print(invert)
    invert_gaussian = invert + gaussian_noise(inputs, mean, std)
    out = 1/invert_gaussian
    return out

if __name__ == "__main__":
    input = torch.tensor([[1, 2], [3, 4], [5, 6]])
    out = invert_gaussian(input, 0, 0.02)
    print("out:")
    print(out)