import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FertilityDataProcessor:

    def __init__(self) -> None:
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        self.parent_directory = os.path.dirname(self.script_directory)
        self.rawGIGACOW_directory = os.path.join(self.parent_directory, 'Data', 'CowData', 'RawGIGACOW')
        os.makedirs(self.rawGIGACOW_directory, exist_ok=True)
        self.rawMESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'RawMESAN')
        os.makedirs(self.rawMESAN_directory, exist_ok=True)

    def load_insemination_data(self, start_date, end_date) -> None:
        logging.info("Loading Insemination Data...")
        insemination_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Insemination.csv")
        insemination_data = pd.read_csv(insemination_data_directory, delimiter=";", low_memory=False)

        insemination_data.drop_duplicates(inplace=True)
        insemination_data['InseminationDate'] = pd.to_datetime(insemination_data['InseminationDate'], errors='coerce')

        col_keep = ["SE_Number", "FarmName_Pseudo", "InseminationDate", "Breeder"]
        insemination_data = insemination_data[col_keep]

        insemination_data = insemination_data[(insemination_data['InseminationDate'] >= start_date) & 
                                              (insemination_data['InseminationDate'] <= end_date)]

        insemination_data = insemination_data.sort_values(by=["SE_Number", "InseminationDate"])

        # Add YearSeason variable
        insemination_data["YearSeason"] = insemination_data["InseminationDate"].apply(lambda x: f"{x.year}-{self.get_season(x.month)}")

        self.fertility_data = insemination_data
        logging.info("Insemination Data Loaded Successfully")

    def get_season(self, month):
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        elif month in [9, 10, 11]:
            return 4

    def load_breed_birth_data(self) -> None:
        logging.info("Loading Breed & Birth Data...")
        breed_data_directory = os.path.join(self.rawGIGACOW_directory, "Cow.csv")
        breed_data = pd.read_csv(breed_data_directory, delimiter=";", low_memory=False)
        
        breed_data.drop_duplicates(inplace=True)
        
        self.fertility_data = pd.merge(self.fertility_data, breed_data[['SE_Number', 'BreedName']], on='SE_Number', how='left')
        self.fertility_data = pd.merge(self.fertility_data, breed_data[['SE_Number', 'BirthDate']], on='SE_Number', how='left')
        self.fertility_data = pd.merge(self.fertility_data, breed_data[['SE_Number', 'Mother']], on='SE_Number', how='left')
        self.fertility_data = pd.merge(self.fertility_data, breed_data[['SE_Number', 'Father']], on='SE_Number', how='left')

        logging.info("Breed & Birth Data Loaded Successfully")

    def load_calving_data(self, start_date, end_date) -> None:
        logging.info("Loading Calving Data...")
        calving_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Calving.csv")
        calving_data = pd.read_csv(calving_data_directory, delimiter=";", low_memory=False)

        calving_data.drop_duplicates(inplace=True)
        calving_data['CalvingDate'] = pd.to_datetime(calving_data['CalvingDate'], errors='coerce')

        calving_data["CalvingEase"] = calving_data["CalvingEase"].replace({
            "1 Normal delivery": "11",
            "2 Difficult delivery": "13",
            "9 Early calving (215-240 days)": "9",
            "11 Easy, without assistance": "11",
            "12 Easy, with assistance": "12",
            "13 Difficult, without veterinary assistance": "13",
            "14 Difficult, with veterinary assistance": "14",
            "15 Not specified": "15",
            "nan": "15",
            "485": "15",
            "486": "15",
            "487": "15",
            "Normal": "11",
            "3 Abnormal position": "3",
            "8 Kastning (<215 dagar)": "8",
            "09 Tidig kalvning": "9",
            "12 Lätt med hjälp": "12",
            "13 Svår utan veterinärhjälp": "13",
            "14 Svår med veterinärhjälp": "14"
        }).astype(str)

        col_keep = ["SE_Number", "CalvingDate", "CalvingSireBullID", "CalvingEase"]
        calving_data = calving_data[col_keep]

        calving_data = calving_data[(calving_data['CalvingDate'] >= start_date) & (calving_data['CalvingDate'] <= end_date)]

        self.fertility_data = pd.merge_asof(
            self.fertility_data.sort_values('InseminationDate'),
            calving_data.sort_values('CalvingDate'),
            left_on='InseminationDate',
            right_on='CalvingDate',
            by='SE_Number',
            direction='forward'
        )

        logging.info("Calving Data Loaded Successfully")

    def load_lactation_data(self) -> None:
        logging.info("Loading Lactation Data...")
        lactation_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Lactation.csv")
        lactation_data = pd.read_csv(lactation_data_directory, delimiter=";", low_memory=False)

        lactation_data.drop_duplicates(inplace=True)
        lactation_data['LactationInfoDate'] = pd.to_datetime(lactation_data['LactationInfoDate'], errors='coerce')
        lactation_data = lactation_data.dropna(subset=['LactationNumber'])

        self.lactation_data = lactation_data
        logging.info("Lactation Data Loaded Successfully")

    def add_lactation_numbers(self) -> None:
        logging.info("Adding Lactation Numbers...")

        if not hasattr(self, 'fertility_data'):
            raise AttributeError("Fertility data has not been loaded. Please run load_insemination_data() first.")
        if not hasattr(self, 'lactation_data'):
            raise AttributeError("Lactation data has not been loaded. Please run load_lactation_data() first.")

        lactation_numbers = []

        grouped = self.fertility_data.groupby('SE_Number')
        for se_number, group in grouped:
            cow_lactation_data = self.lactation_data[self.lactation_data['SE_Number'] == se_number].sort_values('LactationInfoDate')
            if cow_lactation_data.empty:
                continue
            
            lactation_number = None

            for _, row in group.iterrows():
                insemination_date = row['InseminationDate']
                prior_lactation = cow_lactation_data[cow_lactation_data['LactationInfoDate'] <= insemination_date].tail(1)
                if not prior_lactation.empty:
                    lactation_number = prior_lactation['LactationNumber'].values[0]

                lactation_numbers.append({
                    'SE_Number': se_number,
                    'InseminationDate': insemination_date,
                    'LactationNumber': lactation_number
                })

        lactation_df = pd.DataFrame(lactation_numbers)
        self.fertility_data = pd.merge(self.fertility_data, lactation_df, on=['SE_Number', 'InseminationDate'], how='left')

        self.fertility_data['BirthDate'] = pd.to_datetime(self.fertility_data['BirthDate'], errors='coerce')

        def set_initial_lactation(row):
            if pd.isna(row['LactationNumber']):
                birth_date = row['BirthDate']
                insemination_date = row['InseminationDate']
                if birth_date is not None and (insemination_date - birth_date).days / 365.25 <= 2.5:
                    return 1
            return row['LactationNumber']

        self.fertility_data['LactationNumber'] = self.fertility_data.apply(set_initial_lactation, axis=1)

        self.fertility_data.sort_values(by=['SE_Number', 'InseminationDate'], inplace=True)

        def conditional_ffill(group):
            last_lactation_number = None
            for i in range(len(group)):
                if pd.notna(group.iloc[i]['LactationNumber']):
                    last_lactation_number = group.iloc[i]['LactationNumber']
                else:
                    if (group.iloc[:i]['CalvingDate'].isna()).all():
                        group.at[group.index[i], 'LactationNumber'] = last_lactation_number
            return group

        self.fertility_data = self.fertility_data.groupby('SE_Number', group_keys=False).apply(conditional_ffill)

        logging.info("Lactation Numbers Added Successfully")

    def load_calculated_data(self) -> None:
        logging.info("Calculating Derived Data...")

        if not hasattr(self, 'fertility_data'):
            raise AttributeError("Fertility data has not been loaded. Please run load_insemination_data() first.")

        df = self.fertility_data.copy()
        
        df.drop_duplicates(inplace=True)
        df['CalvingDate'] = pd.to_datetime(df['CalvingDate'], errors='coerce')
        df['InseminationDate'] = pd.to_datetime(df['InseminationDate'], errors='coerce')

        df = df[['SE_Number', 'FarmName_Pseudo', 'InseminationDate', 'CalvingDate'] + 
                [col for col in df.columns if col not in ['SE_Number', 'FarmName_Pseudo', 'InseminationDate', 'CalvingDate']]]

        df.sort_values(by=['SE_Number', 'InseminationDate'], inplace=True)

        df['PrevInsemination'] = df.groupby(['SE_Number', 'LactationNumber'])['InseminationDate'].shift(1)
        df['NextInsemination'] = df.groupby(['SE_Number', 'LactationNumber'])['InseminationDate'].shift(-1)

        df['NINS'] = df.groupby(['SE_Number', 'LactationNumber'])['InseminationDate'].transform('nunique')

        df['NextCalving'] = df.groupby(['SE_Number'])['CalvingDate'].shift(-1)

        first_last_insemination = df.groupby(['SE_Number', 'LactationNumber']).agg(
            FirstInsemination=('InseminationDate', 'first'),
            LastInsemination=('InseminationDate', 'last')
        ).reset_index()

        df = pd.merge(df, first_last_insemination, on=['SE_Number', 'LactationNumber'], how='left')

        df['FLI'] = (df['LastInsemination'] - df['FirstInsemination']).dt.days

        df['NextFirstInsemination'] = df.groupby('SE_Number')['FirstInsemination'].shift(-1)
        df['NextLastInsemination'] = df.groupby('SE_Number')['LastInsemination'].shift(-1)

        df['CFI'] = (df['NextFirstInsemination'] - df['CalvingDate']).dt.days
        df['CLI'] = (df['NextLastInsemination'] - df['CalvingDate']).dt.days

        df.loc[df['CFI'] < 0, 'CFI'] = None
        df.loc[df['CLI'] < 0, 'CLI'] = None

        df['GL'] = (df['CalvingDate'] - df['LastInsemination']).dt.days

        df['CI'] = df.groupby('SE_Number')['CalvingDate'].diff().dt.days

        # Set PregnancyCheck: 1 for last insemination in the group, 0 otherwise
        df['PregnancyCheck'] = df.groupby(['SE_Number', 'LactationNumber'])['InseminationDate'].transform(
            lambda x: (x == x.max()).astype(int)
        )

        df.sort_values(by=['SE_Number', 'InseminationDate'], inplace=True)

        self.fertility_data = df
        logging.info("Derived Data Calculated Successfully")


    def weather_preprocessing(self, points: list, start_date: datetime.date, end_date: datetime.date) -> None:
        param = 117
        pname = "Global irradiance"
        interval = "hourly"
        self.MESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'MESAN')
        os.makedirs(self.MESAN_directory, exist_ok=True)
        logging.info("Adding Global Irradiance and THI_adj...")

        pbar = tqdm(points, desc='Adding Global Irradiance and THI_adj ...')
        for point in pbar:
            name = point["id"]
            lat = point["lat"]
            lon = point["lon"]
            pbar.set_description(f'Adding Global Irradiance and THI_adj to {name}')

            fname = f"{name}_2022-2023.csv"
            fpath = os.path.join(self.rawMESAN_directory, fname)

            if not os.path.exists(fpath):
                continue
            df = pd.read_csv(fpath, delimiter=';')

            df.rename(columns={'Temperatur': 'Temperature', 'Relativ fuktighet': 'RelativeHumidity'}, inplace=True)

            sDate = start_date.strftime('%Y-%m-%d')
            eDate = end_date.strftime('%Y-%m-%d')

            api_url = f"https://opendata-download-metanalys.smhi.se/api/category/strang1g/version/1/geotype/point/lon/{round(lon, 6)}/lat/{round(lat, 6)}/parameter/{param}/data.json?from={sDate}&to={eDate}&interval={interval}"
            try:
                response = requests.get(api_url)
                response.raise_for_status()
                tf = pd.DataFrame(response.json())
            except requests.RequestException as e:
                logging.error(f"Error fetching weather data for {name}: {e}")
                continue

            tf = tf.rename(columns={"value": pname, "date_time": "Tid"})
            tf['Tid'] = pd.to_datetime(tf['Tid']).dt.tz_localize(None)
            df['Tid'] = pd.to_datetime(df['Tid'])
            df = pd.merge(df, tf, on="Tid")

            df["THI_adj"] = 4.51 + (0.8 * df["Temperature"]) + (df["RelativeHumidity"] / 100 * (df["Temperature"] - 14.4)) + 46.4 - 1.992 * df["Vindhastighet"] + 0.0068 * df["Global irradiance"]
            df['RelativeHumidity'] = df['RelativeHumidity'] / 100

            df.set_index("Tid", inplace=True)
            max_temp_per_day = df["Temperature"].resample("D").max()
            max_temp_per_day = max_temp_per_day.reset_index()
            max_temp_per_day["HW"] = 0
            max_temp_per_day["cum_HW"] = 0.0

            for _, group_data in max_temp_per_day[max_temp_per_day["Temperature"] >= 25].groupby((max_temp_per_day["Temperature"] < 25).cumsum()):
                if len(group_data) >= 5:
                    max_temp_per_day.loc[group_data.index, "HW"] = 1
                    max_temp_per_day.loc[group_data.index, "cum_HW"] += 1.0
                    A = 1.0
                    if len(group_data) > 5:
                        for i in group_data.index[5:]:
                            A += 1
                            max_temp_per_day.loc[i, "cum_HW"] = A
                    max_temp_per_day.loc[group_data.index[-1] + 1 : group_data.index[-1] + 7, "cum_HW"] = (A - 0.01 * np.exp(np.arange(1, 8) * 0.125 * np.log(100 * A))).round(2)
                    max_temp_per_day.loc[group_data.index[-1] + 1 : group_data.index[-1] + 7, "HW"] = 1

            df["Tid"] = pd.to_datetime(df.index)
            df = pd.merge(df, max_temp_per_day[["Tid", "HW", "cum_HW"]], how="left", left_index=True, right_on="Tid")
            df["HW"] = df["HW"].ffill().astype(int)
            df["cum_HW"] = df["cum_HW"].ffill().astype(int)
            df = df.reset_index(drop=True)

            df['StartDate'] = df['Tid'].dt.date
            df['Temp15Threshold'] = df.groupby('StartDate')['Temperature'].transform(lambda x: (x >= 15).any().astype(int))

            output_file_path = os.path.join(self.MESAN_directory, f"processed_data_{name}.csv")
            df.to_csv(output_file_path, index=False)
        logging.info("Global Irradiance and THI_adj Added Successfully")

    def add_weather_data(self, farms=None) -> None:
        logging.info("Adding Weather Data...")
        files = os.listdir(self.MESAN_directory)

        all_farms = self.fertility_data["FarmName_Pseudo"].unique()
        if farms is not None:
            farms = set(farms)
            farms_to_process = [farm for farm in all_farms if farm in farms]
        else:
            farms_to_process = all_farms

        weather_data_dict = {}
        for farm_name in farms_to_process:
            matching_files = [file for file in files if f"processed_data_{farm_name}.csv" == file.strip('"')]
            if len(matching_files) == 1:
                file_path = os.path.join(self.MESAN_directory, matching_files[0])
                temp_data = pd.read_csv(file_path)
                temp_data["Tid"] = pd.to_datetime(temp_data["Tid"], errors='coerce')
                weather_data_dict[farm_name] = temp_data.set_index("Tid", drop=False)

        logging.info("Merging weather and fertility data...")
        results = []
        for farm_name in tqdm(farms_to_process, desc="Merging weather and fertility data", unit="farm"):
            farm_data = self.process_farm_data(farm_name, weather_data_dict)
            results.append(farm_data)

        logging.info("Concatenating results...")
        self.all_data = pd.concat(results)
        logging.info(f"Total rows after concatenation: {len(self.all_data)}")

        logging.info("Dropping rows with missing InseminationDate...")
        self.all_data = self.all_data.dropna(subset=["InseminationDate"])

        logging.info("Dropping unnecessary columns...")
        cols_to_drop = [col for col in self.all_data.columns if col.endswith('_weather')] + ['Unnamed: 0', 'Tid_x', 'Tid_y', 'DateOnly',
                                                                                             'Daggpunktstemperatur', 'Vindhastighet', 
                                                                                             'Vindriktning', 'Nederbörd', 'Snö', 
                                                                                             'Lufttryck', 'Sikt', 'Global irradiance', 
                                                                                             'Nederbördstyp', 'Molnighet', 'Byvind',
                                                                                             'StartDate', 'Tid']
        self.all_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        logging.info("Renaming columns to reflect mean values...")
        self.all_data.rename(columns={
            'Temperature': 'MeanTemperature', 
            'RelativeHumidity': 'MeanRelativeHumidity', 
            'THI_adj': 'MeanTHI_adj',
            'Temp15Threshold': 'MaxTemp15Threshold'
        }, inplace=True)

        logging.info("Saving the final merged data to CSV...")
        output_file_path = os.path.join(self.parent_directory, 'Data', f"MergedData/TheFertilityData.csv")
        self.all_data.to_csv(output_file_path, index=False)
        logging.info("Weather Data Added Successfully")

    def process_farm_data(self, farm_name, weather_data):
        farm_data = self.fertility_data[self.fertility_data["FarmName_Pseudo"] == farm_name].copy()
        if farm_name in weather_data:
            weather = weather_data[farm_name]

            daily_weather = weather.resample('D').agg({
                'Temperature': 'mean',
                'RelativeHumidity': 'mean',
                'THI_adj': 'mean',
                'HW': 'max',
                'cum_HW': 'max',
                'Temp15Threshold': 'max'
            }).reset_index()

            daily_weather.rename(columns={
                'Temperature': 'MeanTemperature',
                'RelativeHumidity': 'MeanRelativeHumidity',
                'THI_adj': 'MeanTHI_adj'
            }, inplace=True)

            farm_data['DateOnly'] = farm_data['InseminationDate'].dt.date
            daily_weather['DateOnly'] = daily_weather['Tid'].dt.date

            farm_data = pd.merge(farm_data, daily_weather, on='DateOnly', how='left')

        else:
            logging.warning(f"No matching weather data found for farm {farm_name}")
        return farm_data
    
    def preprocess(self, start_date, end_date, farms=None):
        self.load_insemination_data(start_date, end_date)
        self.load_breed_birth_data()
        self.load_calving_data(start_date, end_date)
        self.load_lactation_data()
        self.add_lactation_numbers()
        self.load_calculated_data()

        coord_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'Coordinates', 'Coordinates.csv')
        coord_df = pd.read_csv(coord_directory)
        coord_df = coord_df.dropna(subset=['FarmID'])
        coord_df[['lat', 'lon']] = coord_df['Koordinater'].str.split(', ', expand=True).astype(float)
        points = []
        for _, farm in coord_df.iterrows():
            input_point = {'id': farm['FarmID'], 'lat': farm['lat'], 'lon': farm['lon']}
            points.append(input_point)
        start_datetime = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        startdate = start_datetime.date()
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        enddate = end_datetime.date()
        self.weather_preprocessing(points, startdate, enddate)
        self.add_weather_data(farms=farms)
        logging.info('Preprocessing Done!')

def main():
    processor = FertilityDataProcessor()
    start_date = '2022-01-01 00:00:00'
    end_date = '2023-11-13 23:00:00'
    processor.preprocess(start_date=start_date, end_date=end_date, farms=None)

if __name__ == "__main__":
    main()
