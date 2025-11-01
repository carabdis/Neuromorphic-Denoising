# Neuromorphic-Denoising
This repository provide the codes that generate the figures in our article *Neuromorphic Denoising With Fully-Analog Memristive In-Memory Computing*. The contents in the repository are listed below.

## Analog_Memory_DEMO
This folder contains the python code for neuromorphic system control. The connection between the personal computer and the proposed system is demonstrated by the aardvark I2C module. Three kinds of demonstration is provided, including 1D-CNN for Epilpsey Detection, VAE for MNIST and 3D-Shapes picture generation.

## Analog_Memory_System_2023_DE0
This folder is the project file of Quartus 18.1 for Intel DE-V0. The name of top module is the same as that of the project. This project realizes the communication with PC and control of the PCB board presented in the article Fig.5a.

## Architectural_Simulation
This python file is the computation process of Fig.5b. It computes the number of temporal buffers and processing units, estimating the power cost and space cost according to reported parameters of previous works, as well as the synthesis results of Synopsis DC compiler.

## Fig_Plot
This python file provide the code for figure drawing in the article.