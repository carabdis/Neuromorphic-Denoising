import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import operator

total_sum = np.load('ckpt/total_sum.npy')  # 2879*32-10,16

for i in range(16):
    fig = sns.distplot(total_sum[:,i], rug=False, kde = False, hist=True, color="b")

    # fig = sns.kdeplot(sum_inter[:,ch], color="b")
    plt.show()