# Reproduction of the FPDGD algorithm

This repo contains my reproduction and experiment code for the FPDGD algorithm. This is my final project for COMPSCI 660 Advanced Information Assurance (Fall '22) at UMass Amherst.

Please make sure to install all of the following dependencies:
1. numpy, scipy (For all the math)
2. pandas (For the data)
3. sklearn (For attack experiment)
4. tqdm (Nice progress bar)
5. Matplotlib (For plotting)
6. Optional: mypy (For type checking)

To train a model:
1. Download the [LETOR 4.0](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/letor-4-0/) dataset and/or the [MSLR](https://www.microsoft.com/en-us/research/project/mslr/) dataset.
2. Unzip them in the `data` folder.
3. Run `python train.py` (please modify the parameters in this script)
4. The model parameters are saved to the `models` folder, and the plots are saved to the `plots` folder.

To run the privacy attack experiment:
1. Download data and unzip (like above)
2. Run `python mia.py` (please modify the parameters in this script)
