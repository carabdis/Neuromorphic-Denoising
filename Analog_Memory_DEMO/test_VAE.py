import torch
import torch.nn as nn
import pytorch_model as model
import h5py as hpy
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pytorch_utils import noise, display

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}


def main():
    Episodic = torch.load("./MyModel/checkpoint_4454.ptr", map_location=torch.device("cpu"))
    # Episodic_ref = torch.load("./MyModel/checkpoint.ptr", map_location=torch.device("cpu"))
    # encoder = model.Encoder(input_shape=3, latent_dim=20)
    # decoder = model.Decoder(latent_dim=20)
    # Episodic = model.VAE(encoder, decoder)
    # RefDict = Episodic_ref.state_dict()
    # CorrectDict = dict()
    # for that in RefDict:
    #     if that == "encoder.conv1.weight":
    #         CorrectDict["encoder.conv1.layer.weight"] = RefDict[that]
    #     elif that == "encoder.conv1.bias":
    #         CorrectDict["encoder.conv1.layer.bias"] = RefDict[that]
    #     else:
    #         CorrectDict[that] = RefDict[that]
    # CorrectDict["encoder.conv1.InMax"] = Episodic.encoder.conv1.InMax
    # CorrectDict["encoder.conv1.InMin"] = Episodic.encoder.conv1.InMin
    # Episodic.load_state_dict(CorrectDict)
    dataset = hpy.File("./data/3dshapes.h5", "r")
    dataset_images = dataset["images"]
    dataset_labels = dataset["labels"]
    IndexFile = open("./IndexFile.txt", "r")
    Index = IndexFile.read().split('\n')[:-1]
    Index = [int(i) for i in Index]
    train_images, test_images, train_labels, test_labels = train_test_split(dataset_images[:20, :, :, :], dataset_labels[:20, :], random_state=42)
    # print("Filtered Training Images Shape:", train_images.shape)
    # print("Filtered Testing Images Shape:", test_images.shape)
    test_ds = test_images
    np.random.shuffle(test_ds)
    # x_test_new = np.array([add_noise(image) for image in test_ds[0:20]])
    train_images = torch.tensor(np.transpose(train_images, (0, 3, 1, 2))).float()
    noisy_test_numpy = noise(test_ds / 255., noise_factor=0.1)
    noisy_test = torch.tensor(np.transpose(noisy_test_numpy, (0, 3, 1, 2))).float()
    # noisy_test = torch.tensor(np.transpose(test_ds / 255., (0, 3, 1, 2))).float()
    origin, decoded_imgs, mean, var = Episodic(noisy_test)
    decoded_imgs_numpy = np.transpose(decoded_imgs.detach().numpy(), (0, 2, 3, 1))
    # plot_zoom_rows(encoder, decoder, test_ds, 0)
    fig = display(noisy_test_numpy, decoded_imgs_numpy, title='Inputs and outputs for VAE')
    return


if __name__ == "__main__":
    main()