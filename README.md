# Evaluating Resilience to Heat Stress among Dairy Cows in Sweden 🐄🌡️

**This GitHub repository houses the codebase for a study investigating the impact of Swedish weather conditions, particularly heat, on dairy cows' milk production on Swedish farms. Using weather data sourced from Sveriges meteorologiska och hydrologiska institut (SMHI) and extensive dairy data from the Gigacow project at Sveriges lantbruksuniversitet (SLU).**

## Introduction

This project studies the relationship between weather conditions and dairy cow milk production in Swedish farms. The motivation stems from the critical importance of understanding how varying temperatures, specifically heat, influence this aspect of agriculture. By combining data from weather and dairy sources, the study employs a diverse set of mathematical and machine learning techniques. These methods, ranging from normalization techniques to modeling and statistical frameworks, enables a exploration of the dynamics. This GitHub repository serves as a hub for the codebase, providing a foundation for future studies. The report can be found [HERE](https://github.com/jockepolis/HeatStressEvaluation/tree/5ecdee8946b3b22e76e41ee09d8631f3338f92ee/Report/HeatStressEvaluation.pdf).

## Contributors
### Authors
- [Joakim Svensson](https://www.linkedin.com/in/joakim-svensson1998/)
- [Axel Englund](www.linkedin.com/in/axel-englund-826714183)

### Supervisors
- Lena-Mari Tamminen
- Tomas Klingström
- Martin Johnsson


## Features

- Data preprocessing of dairy and weather data.
- Employment of several statistical methods.

## Repo structure
```
HeatStressEvaluation (project-root)/
|-- Data/
|   |-- TheData.csv
|   |
|   |-- CowData/
|   |    |-- CowData_README.md
|   |    |-- GIGACOW/
|   |    |   |-- Cow_filtered.csv
|   |    |   |-- DiagnosisTreatment_filtered.csv
|   |    |   |-- Lactation_filtered.csv
|   |    |   |-- MilkYield_filtered.csv
|   |    |   |-- Robot_filtered.csv
|   |    |
|   |    |-- RawGIGACOW/
|   |        |-- Cow.csv
|   |        |-- DiagnosisTreatment.csv
|   |        |-- Lactation.csv
|   |        |-- MilkYield.csv
|   |        |-- Robot.csv
|   |
|   |-- WeatherData/
|       |-- WeatherData_README.md
|       |-- Coordinates/
|       |   |-- Coordinates.csv
|       |
|       |-- MESAN/
|       |   |-- processed_data_XXXX.csv
|       |   |   ...
|       |   |-- ...
|       |
|       |-- RawMESAN/
|           |-- XXXX_2022-2023.csv
|           |   ...
|           |-- ...
|       
|-- DataPreprocessing/
|   |-- Preprocesses.py
|   |-- DataPreprocessing.ipynb
|
|-- Modeling/
|   |-- Bayesian.py
|   |-- BayesianGAM.ipynb
|   |-- BayesianLinear.ipnyb
|   |-- DataExploration.ipynb
|   |-- DIMReduction.ipynb
|   |-- RandomForest.ipynb
|   |-- ShortBreedStudy.ipynb
|   |-- BoxPlots.ipynb
|
|-- README.md
|-- requirements.txt
```
## Prerequisites
Before running the scripts, make sure to fulfill the following prerequisites:
### 0. Git Clone
```bash
git clone https://github.com/axeUUeng/HeatStressEvaluation.git
```
And then change into the project directory:
```bash
cd /path/to/HeatStressEvaluation
```
Replace `/path/to/HeatStressEvaluation` with the actual path to the `HeatStressEvaluation` project directory.
### 1. Installation
Follow one of the installation guides for conda
- [Anaconda Installation](https://docs.anaconda.com/free/anaconda/install/index.html)
- [Miniconda Installation](https://docs.conda.io/projects/miniconda/en/latest/)

Python version used by the authors is `3.10.13`.

Then to get the proper environment:
```bash
# Conda env installation command
conda create --name your_environment_name --file requirements.txt
```
If the creation of the environment doesn't work for some reason, the most important libraries are:
- `Numpy`
- `requests`
- `scikit-learn`
- `Numba`
- `matplotlib`
- `Seaborn`
- `Pandas`
- `SciPy`
- `Patsy`
- `tqdm`
- `statsmodels`
- `Umap`
- `itertools`

### 2. Datasets

Some datasets are necessary and should be placed in the "Data" folder according to the structure provided above. Ensure the availability of the following datasets and their correct placement:

- The Gigacow data from SLU in the `Data/CowData/RawGIGACOW/` directory
    - `Cow.csv`
    - `DiagnosisTreatment.csv`
    - `Lactation.csv`
    - `MilkYield.csv`
    - `Robot.csv`
- The MESAN data from SMHI in the `Data/WeatherData/RawMESAN/` directory
    - `XXXX_2022-2023.csv`
- The coordinate file in the `Data/WeatherData/Coordinates/` directory 
    - `Coordinates.csv`
### 2. Preproccess the data
Run the two cells in `DataPreprocessing/DataPreprocessing.ipynb`.

Resulting dataset with milk-records merged with weather is named and stored under `Data/TheData.csv`.
### 3. Run the models in `/Modeling/`

## `/Modeling/` Content
- `DataExploration.ipynb` - contains some initial exploration of the data, mainly focusing on the number of records for each farm.
- `Bayesian.py` - contains scripts and functions used in `BayesianLinear.ipynb`
- `BayesianLinear.ipynb` - fits a linear combinations of features to normalized daily total yield. For one cow, all cows on one farm and one model for one farm.
- `BayesianGAM.ipynb` - fits a GAM model to either one farm or one cow.
- `BoxPlots.ipynb` - shows basic differences in temperature and yield for mainly summer 22 and summer 23.
- `DIMReduction.ipynb` - short attempt at dimension reductions on the dataset.
- `ShortBreedStudy.ipynb` - short attempt at visualising differences between breeds in yield during `HW=1` and `HW=0`.
- `RandomForest.ipynb` - Applies normalization to yield and then uses RandomForests to find patterns in the data.
