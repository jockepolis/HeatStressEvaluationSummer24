# Evaluating Resilience to Heat Stress among Dairy Cows in Sweden 🐄🌡️ - Summer of 2024

**This GitHub repository houses the codebase for a study investigating the impact of Swedish weather conditions, particularly heat, on dairy cows' milk production on Swedish farms. Using weather data sourced from Sveriges meteorologiska och hydrologiska institut (SMHI) and extensive dairy data from the Gigacow project at Sveriges lantbruksuniversitet (SLU).**

## Introduction

(This project studied the relationship between weather conditions and dairy cow milk production in Swedish farms. The motivation stems from the critical importance of understanding how varying temperatures, specifically heat, influence this aspect of agriculture. By combining data from weather and dairy sources, the study employs a diverse set of mathematical and machine learning techniques. These methods, ranging from normalization techniques to modeling and statistical frameworks, enables a exploration of the dynamics. This GitHub repository serves as a hub for the codebase, providing a foundation for future studies. The report, made by Axel Englund & Joakim Svensson made the winter of 2023/2024 can be found [HERE](https://github.com/jockepolis/HeatStressEvaluation/tree/5ecdee8946b3b22e76e41ee09d8631f3338f92ee/Report/HeatStressEvaluation.pdf).)

During the summer of 2024, me (Joakim) continued on this project and developed larger models for more data etc.

## Contributors
### Authors
- [Joakim Svensson](https://www.linkedin.com/in/joakim-svensson1998/)

### Supervisors
- Lena-Mari Tamminen
- Tomas Klingström
- Martin Johnsson
- Patricia Ask Gullstrand


## Features

- Data preprocessing of dairy and weather data.
- Employment of several statistical methods.

## Lathund: How to Run the Models

Follow these steps to successfully run the models in this repo. Make sure you complete each step carefully to ensure smooth execution.

### Step 1: Prepare Raw Data

- **Ensure you have the required raw data:**
  - Place the **GIGACOW data** in the `RawGIGACOW` folder.
  - Place the **weather data** in the `RawMESAN` folder.
  
- **Verify data consistency:**
  - Check that the filenames of the raw data match the names expected in the different preprocessing scripts.
  - Confirm that the dates and times align with the `start_date` and `end_date` in the preprocessing files.

- **Run the data preprocessing:**
  - Navigate to the `DataPreProcessing` folder and run the `DataPreProcessing.ipynb` notebook.
  - After running, three new CSV files containing the preprocessed data will be automatically created and saved in the `MergedData` folder.

### Step 2: Clean the Data

- **Run the DataCleaning scripts:**
  - Execute the data cleaning scripts in the corresponding folder.
  - These scripts perform various data checks, imputations, and cleaning tasks.
  
- **Troubleshoot if needed:**
  - If something doesn't work, check the error messages. It might be a missing data file, a mismatch in file naming, or another small issue.

- **Check output:**
  - After running the cleaning scripts, new CSV files will be generated in the `MergedData` folder, with names like `CleanxxxxxData.csv`.

  **Note:** If I were you, I would skip the fertility and calving data at this stage. Although they have been cleaned and combined with the weather data, we did not have time to model those, as our focus was more on yield.

### Step 3: Run the Wilminks Scripts

- **Run all three scripts in the `Wilminks` folder:**
  - This will generate the following datasets:
    - `HeatApproachYieldDataTest.csv`
    - `MilkApproachYieldDataTest.csv`
    - `HeatApproachYieldDataTestQuantile.csv`
    
- **Proceed to modeling:**
  - Once these datasets are created, you're ready to run the models.
  
### Step 4: Running the Models

- **Model scripts:**
  - The models are located in two folders:
    - `ModelingFarmLevelYield` (for farm-level models)
    - `ModelingIndividualsYield` (for individual-level models)

- **Recommended approach:**
  - According to our experience, the **HeatApproach models** are the best.
    - Refer to the `Sammanställ av resultat` document to understand why.
    - The **GAM HeatApproach** model is the top performer for both farm-level and individual-level yield predictions.
  
- **Choosing datasets:**
  - When running the HeatApproach models, you can choose between the regular Wilminks data and the Quantile Wilminks data. 
  - Simply select the desired dataset by loading one of the following files in the script:
    - Regular data: `HeatApproachYieldDataTest.csv`
    - Quantile data: `HeatApproachYieldDataTestQuantile.csv`
  - In each and every script for the models, it starts with loadin the data and there you can change the name of the desired file.
  
  Example in the script:
```python
  milk_data = pd.read_csv('../Data/MergedData/HeatApproachYieldDataTest.csv', dtype=dtype_dict)
```
  
## Repo structure
```

