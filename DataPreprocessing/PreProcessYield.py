import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import requests
import os
from joblib import Parallel, delayed
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MilkDataProcessor:
    def __init__(self) -> None:
        self.script_directory = os.path.dirname(os.path.realpath(__file__))
        self.parent_directory = os.path.dirname(self.script_directory)
        self.rawGIGACOW_directory = os.path.join(self.parent_directory, 'Data', 'CowData', 'RawGIGACOW')
        os.makedirs(self.rawGIGACOW_directory, exist_ok=True)
        self.rawMESAN_directory = os.path.join(self.parent_directory, 'Data', 'WeatherData', 'RawMESAN')
        os.makedirs(self.rawMESAN_directory, exist_ok=True)
    
    def load_milk_data(self, start_date='2022-01-01', end_date='2024-08-18') -> None:
        logging.info("Loading Milk Data...")
        milk_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_CowMilkYield_Common240823.csv")
        try:
            milk = pd.read_csv(milk_data_directory, delimiter=";", low_memory=False)
        except FileNotFoundError:
            logging.error(f"File not found: {milk_data_directory}")
            return
        
        milk_data = milk.drop(columns=[
            "Del_Cow_Id", "LactationInfoSource", "OriginalFileSource", 
            "dwh_factCowMilk_Id", "dwh_factCowMilkOther_Id"
        ])
        milk_data["TotalYield"] = pd.to_numeric(milk_data["TotalYield"].str.replace(",", "."), errors="coerce")
        milk_data["StartDate"] = pd.to_datetime(milk_data["StartDate"], errors='coerce')
        milk_data["StartTime"] = pd.to_datetime(milk_data["StartTime"], format='%H:%M:%S', errors='coerce').dt.time

        mask_unknown_se = (milk_data['SE_Number'] == 'Unknown') & (milk_data['AnimalNumber'] != -1)
        milk_data.loc[mask_unknown_se, 'SE_Number'] = (
            'SE-' + milk_data.loc[mask_unknown_se, 'FarmName_Pseudo'] + '-' + milk_data.loc[mask_unknown_se, 'AnimalNumber'].astype(str)
        )

        milk_data = milk_data[(milk_data['StartDate'] >= start_date) & (milk_data['StartDate'] <= end_date)]
        milk_data["DateTime"] = pd.to_datetime(milk_data["StartDate"].astype(str) + ' ' + milk_data["StartTime"].astype(str), errors='coerce')
        milk_data["DateOnly"] = milk_data["StartDate"]

        # Add YearSeason variable
        milk_data["YearSeason"] = milk_data["StartDate"].apply(lambda x: f"{x.year}-{self.get_season(x.month)}")

        self.milk_data = milk_data
        self.load_breed_birth_data(start_date, end_date)
        logging.info("Milk Data Loaded Successfully")
    
    def get_season(self, month):
        if month in [12, 1, 2]:
            return 1
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        elif month in [9, 10, 11]:
            return 4

    def load_breed_birth_data(self, start_date='2022-01-01', end_date='2024-08-18') -> None:
        logging.info("Loading Breed & Birth Data...")
        breed_data_directory = os.path.join(self.rawGIGACOW_directory, "Del_Cow240823.csv")
        try:
            breed_data = pd.read_csv(breed_data_directory, delimiter=";", low_memory=False)
        except FileNotFoundError:
            logging.error(f"File not found: {breed_data_directory}")
            return

        self.milk_data = pd.merge(self.milk_data, breed_data[['SE_Number', 'BreedName']], on='SE_Number', how='left')
        self.milk_data = pd.merge(self.milk_data, breed_data[['SE_Number', 'BirthDate']], on='SE_Number', how='left')
        self.milk_data = pd.merge(self.milk_data, breed_data[['SE_Number', 'Mother']], on='SE_Number', how='left')
        self.milk_data = pd.merge(self.milk_data, breed_data[['SE_Number', 'Father']], on='SE_Number', how='left')

        breed_data['CullDecisionDate'] = pd.to_datetime(breed_data['CullDecisionDate'], errors='coerce')
        valid_cull_decision_dates = breed_data[(breed_data['CullDecisionDate'] >= start_date) & 
                                               (breed_data['CullDecisionDate'] <= end_date)]
        self.milk_data = pd.merge(self.milk_data, valid_cull_decision_dates[['SE_Number', 'CullDecisionDate']], on='SE_Number', how='left')

        logging.info(f"Rows after merging breed & birth data: {len(self.milk_data)}")
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

            df["THI_adj"] = 4.51 + (0.8 * df["Temperatur"]) + (df["Relativ fuktighet"] / 100 * (df["Temperatur"] - 14.4)) + 46.4 - 1.992 * df["Vindhastighet"] + 0.0068 * df["Global irradiance"]
            df['Relativ fuktighet'] = df['Relativ fuktighet'] / 100

            df.set_index("Tid", inplace=True)
            max_temp_per_day = df["Temperatur"].resample("D").max()
            max_temp_per_day = max_temp_per_day.reset_index()
            max_temp_per_day["HW"] = 0
            max_temp_per_day["cum_HW"] = 0.0

            for _, group_data in max_temp_per_day[max_temp_per_day["Temperatur"] >= 25].groupby((max_temp_per_day["Temperatur"] < 25).cumsum()):
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
            df['Temp15Threshold'] = df.groupby('StartDate')['Temperatur'].transform(lambda x: (x >= 15).any().astype(int))

            output_file_path = os.path.join(self.MESAN_directory, f"processed_data_{name}.csv")
            df.to_csv(output_file_path, index=False)
        logging.info("Global Irradiance and THI_adj Added Successfully")
    
    def process_farm_data(self, farm_name, weather_data):
        farm_data = self.milk_data[self.milk_data["FarmName_Pseudo"] == farm_name].copy()
        if farm_name in weather_data:
            weather = weather_data[farm_name]
            farm_data["DateHour"] = farm_data["DateTime"].dt.floor("h")
            farm_data["DateOnly"] = pd.to_datetime(farm_data["DateOnly"])

            if 'Tid' in weather.columns:
                weather_data_by_day = weather.resample('D').mean(numeric_only=True)
                weather_data_by_day["Tid"] = weather_data_by_day.index

                farm_data = farm_data.set_index("DateOnly").join(weather_data_by_day.set_index("Tid"), rsuffix="_weather", on="DateOnly").reset_index()
                farm_data["DateHour"] = farm_data["DateTime"].dt.floor("h")

                farm_data = farm_data.set_index("DateHour").join(weather.set_index("Tid"), rsuffix="_weather", on="DateHour").reset_index()
                farm_data.drop("DateHour", axis=1, inplace=True)
            else:
                logging.warning(f"Weather data for farm {farm_name} does not contain 'Tid' column.")
        else:
            logging.warning(f"No matching weather data found for farm {farm_name}")
        return farm_data

    def add_weather_data(self, farms=None) -> None:
        logging.info("Adding Weather Data...")
        files = os.listdir(self.MESAN_directory)

        all_farms = self.milk_data["FarmName_Pseudo"].unique()
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

        logging.info("Merging weather and milk data...")
        results = Parallel(n_jobs=-1)(delayed(self.process_farm_data)(farm_name, weather_data_dict) for farm_name in tqdm(farms_to_process, desc="Merging weather and milk data", unit="farm"))

        logging.info("Concatenating results...")
        self.all_data = pd.concat(results)
        logging.info(f"Total rows after concatenation: {len(self.all_data)}")

        logging.info("Dropping rows with missing TotalYield...")
        self.all_data = self.all_data.dropna(subset=["TotalYield"])

        logging.info("Dropping unnecessary columns...")
        cols_to_drop = [col for col in self.all_data.columns if col.endswith('_weather')] + ['Unnamed: 0', 'Tid_x', 'Tid_y', 'DateOnly',
                                                                                             'Daggpunktstemperatur', 'Vindhastighet', 
                                                                                             'Vindriktning', 'Nederbörd', 'Snö', 
                                                                                             'Lufttryck', 'Sikt', 'Global irradiance', 
                                                                                             'Nederbördstyp', 'Molnighet', 'Byvind', 
                                                                                              'SessionNumber']
        self.all_data.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        logging.info("Renaming columns...")
        self.all_data.rename(columns={'Temperatur': 'Temperature', 'Relativ fuktighet': 'RelativeHumidity'}, inplace=True)

        logging.info("Saving the final merged data to CSV...")
        output_file_path = os.path.join(self.parent_directory, 'Data', f"MergedData/TheYieldData.csv")
        self.all_data.to_csv(output_file_path, index=False)
        logging.info("Weather Data Added Successfully")

    def preprocess(self, start_date, end_date, farms=None):
        self.load_milk_data(start_date=start_date.split()[0], end_date=end_date.split()[0])
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
    processor = MilkDataProcessor()
    start_date = '2022-01-01 00:00:00'
    end_date = '2024-08-18 23:00:00'
    processor.preprocess(start_date=start_date, end_date=end_date, farms=None)

if __name__ == "__main__":
    main()
