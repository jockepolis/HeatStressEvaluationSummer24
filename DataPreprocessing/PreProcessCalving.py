import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import requests
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CalvingDataProcessor:

    def __init__(self) -> None:
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        self.parent_directory = os.path.dirname(self.script_directory)
        self.rawGIGACOW_directory = os.path.join(self.parent_directory, 'Data', 'CowData', 'RawGIGACOW')
        os.makedirs(self.rawGIGACOW_directory, exist_ok=True)
        self.rawMESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'RawMESAN')
        os.makedirs(self.rawMESAN_directory, exist_ok=True)

    def load_calving_data(self, start_date, end_date) -> None:
        logging.info("Loading Calving Data...")
        calving_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Calving240823.csv")
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
            "nan": "15"
        }).astype(str)

        col_keep = ["SE_Number", "FarmName_Pseudo", "CalvingDate", "CalvingSireBullID", "CalvingEase"]
        calving_data = calving_data[col_keep]

        calving_data = calving_data[(calving_data['CalvingDate'] >= start_date) & 
                                    (calving_data['CalvingDate'] <= end_date)]

        calving_data = calving_data.sort_values(by=["SE_Number", "CalvingDate"])

        # Add YearSeason variable
        calving_data["YearSeason"] = calving_data["CalvingDate"].apply(lambda x: f"{x.year}-{self.get_season(x.month)}")

        self.calving_data = calving_data
        logging.info("Calving Data Loaded Successfully")

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
        breed_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Cow240823.csv")
        breed_data = pd.read_csv(breed_data_directory, delimiter=";", low_memory=False)
        
        breed_data.drop_duplicates(inplace=True)
        
        self.calving_data = pd.merge(self.calving_data, breed_data[['SE_Number', 'BreedName', 'BirthDate', 'Mother', 'Father']], 
                                     on='SE_Number', how='left')

        logging.info("Breed & Birth Data Loaded Successfully")

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

            fname = f"{name}_2022-2024.csv"
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

        all_farms = self.calving_data["FarmName_Pseudo"].unique()
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

        logging.info("Merging weather and calving data...")
        results = []
        for farm_name in tqdm(farms_to_process, desc="Merging weather and calving data", unit="farm"):
            farm_data = self.process_farm_data(farm_name, weather_data_dict)
            results.append(farm_data)

        logging.info("Concatenating results...")
        self.all_data = pd.concat(results)
        logging.info(f"Total rows after concatenation: {len(self.all_data)}")

        logging.info("Dropping rows with missing CalvingDate...")
        self.all_data = self.all_data.dropna(subset=["CalvingDate"])

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
        output_file_path = os.path.join(self.parent_directory, 'Data', f"MergedData/TheCalvingData.csv")
        self.all_data.to_csv(output_file_path, index=False)
        logging.info("Weather Data Added Successfully")

    def process_farm_data(self, farm_name, weather_data):
        farm_data = self.calving_data[self.calving_data["FarmName_Pseudo"] == farm_name].copy()
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

            farm_data['DateOnly'] = farm_data['CalvingDate'].dt.date
            daily_weather['DateOnly'] = daily_weather['Tid'].dt.date

            farm_data = pd.merge(farm_data, daily_weather, on='DateOnly', how='left')

        else:
            logging.warning(f"No matching weather data found for farm {farm_name}")
        return farm_data
    
    def preprocess(self, start_date, end_date, farms=None):
        self.load_calving_data(start_date, end_date)
        self.load_breed_birth_data()

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
    processor = CalvingDataProcessor()
    start_date = '2022-01-01 00:00:00'
    end_date = '2024-08-18 23:00:00'
    processor.preprocess(start_date=start_date, end_date=end_date, farms=None)

if __name__ == "__main__":
    main()
