# ⛈️Weather Data☀️

Welcome to the `WeatherData` folder! This directory contains the weather datasets used in this project. Collected from [SMHI](https://www.smhi.se/data/oppna-data/meteorologiska-data/analysmodell-mesan-1.30445).

## Datasets

### 1. `/Coordinates/`

- **Description**: This dataset contains information about the farm coordinates and their pseudo farm IDs.
- **File**:
  - `Coordinates.csv`: The only file in this folder.
    - Needs the following structure:
    ```csv
    Koordinater,FarmID,
    "00.00000, 10.00000", XXXXX,
    "01.00000, 11.00000",YYYYY,
    ```

### 2. `/RawMESAN/`
- **Description**: Raw data from SMHI, XXXX is the farm pseudos.
- **File(s)**:
  - `XXXX_2022-2023.csv`: Weather data in csv format.
    - Has the following columns:
    ```csv
    Tid;Temperatur;Daggpunktstemperatur;Relativ fuktighet;Vindhastighet;Vindriktning;Byvind;Nederbörd;Snö;Nederbördstyp;Molnighet;Sikt;Lufttryck
    ```


### 3. `/MESAN/`
- **Description**: Preprocessed MESAN data (from `/RawMESAN/`), addition of `Global Irradiance` and `THI_adj`, fetched from [STRÅNG](https://www.smhi.se/forskning/forskningsenheter/meteorologi/strang-en-modell-for-solstralning-1.329). Will be merged with GIGACOW-data.
- **File(s)**:
  - `processed_data_XXXX.csv`: Data in csv format.
    - Has the following columns:
    ```csv
    Tid,Temperatur,Daggpunktstemperatur,Relativ fuktighet,Vindhastighet,Vindriktning,Byvind,Nederbörd,Snö,Nederbördstyp,Molnighet,Sikt,Lufttryck,Global irradiance,THI_adj
    ```

## Notes

- Columns are in Swedish here, is in English in `TheData.csv`.
- Ordered from SMHI.

Happy co(-w)ding! 🐮💻
