import os
from glob import glob
from typing import Sequence

import numpy as np
import pandas as pd
import wget

bikaner_coords = [28., 73.3]
jodhpur_coords = [26.3, 73.017]
seattle = [47.4444, -122.3139]
portland = [45.5908, -122.6003]


def download_ghcn_daily_india(years: Sequence):
    path_to_directory = os.getenv("INDIA_DATA") or './data'
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5],
                             header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    stations = stations[((stations.LATITUDE == bikaner_coords[0]) & (stations.LONGITUDE == bikaner_coords[-1]))
                        | ((stations.LATITUDE == jodhpur_coords[0]) & (stations.LONGITUDE == jodhpur_coords[-1]))]
    for year in np.arange(*years, step=1):
        target_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"
        if not os.path.isfile(f"{path_to_directory}/{year}.csv.gz"):
            try:
                print(f"Downloading url {target_url} to file {path_to_directory}")
                wget.download(target_url, path_to_directory)
                data = pd.read_csv(f"{path_to_directory}/{year}.csv.gz", compression='gzip', header=None,
                                   names=["STATION", "DATE", "ELEMENT", "VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "TIME"])
                data = data[(data['ELEMENT'].isin(['TMAX']))].rename(columns={'VALUE': 'TMAX'}).merge(stations,
                                                                                                      on='STATION').drop(
                    columns=['ELEMENT'])
                data = data.assign(DATE=pd.to_datetime(data.DATE.astype(str)))
                # drop stations for this year when >= 70% unvalid May-June data and drop data with at least 1 quality flag
                data = data[data['Q-FLAG'].isna()].assign(YEAR=data.DATE.dt.year).assign(MONTH=data.DATE.dt.month)
                count_station_year = data[(data['MONTH'] >= 5) & (data['MONTH'] <= 6)].groupby(['STATION']).TMAX.count()
                valid_stations = count_station_year[count_station_year >= 0.7 * 61].index
                data = data.set_index(['STATION', 'DATE', 'YEAR']).loc[valid_stations][
                    ['TMAX', 'TIME', 'STATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
                if len(data):
                    data.to_csv(f"{path_to_directory}/{year}.csv")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def download_ghcn_daily_portland_seattle(years: Sequence):
    path_to_directory = os.getenv("CANADA_DATA") or './data'
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5],
                             header=None,
                             names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    stations = stations[((stations.LATITUDE == seattle[0]) & (stations.LONGITUDE == seattle[-1]))
                        | ((stations.LATITUDE == portland[0]) & (stations.LONGITUDE == portland[-1]))]
    for year in np.arange(*years, step=1):
        target_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"
        if not os.path.isfile(f"{path_to_directory}/{year}.csv.gz"):
            try:
                print(f"Downloading url {target_url} to file {path_to_directory}")
                wget.download(target_url, path_to_directory)
                data = pd.read_csv(f"{path_to_directory}/{year}.csv.gz", compression='gzip', header=None,
                                   names=["STATION", "DATE", "ELEMENT", "VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "TIME"])
                data = data[(data['ELEMENT'].isin(['TMAX']))].rename(columns={'VALUE': 'TMAX'}).merge(stations,
                                                                                                      on='STATION').drop(
                    columns=['ELEMENT'])
                data = data.assign(DATE=pd.to_datetime(data.DATE.astype(str)))
                data = data[data['Q-FLAG'].isna()].assign(YEAR=data.DATE.dt.year).assign(MONTH=data.DATE.dt.month)
                data = data.set_index(['STATION', 'DATE', 'YEAR'])[
                    ['TMAX', 'TIME', 'STATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
                if len(data):
                    data.to_csv(f"{path_to_directory}/{year}.csv")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def get_ghcn_daily_india_annualmax():
    path_to_directory = os.getenv('INDIA_DATA')
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")] if path_to_directory is not None else [pd.read_csv("./data/phalodi.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(
        TMAX=lambda x: x['TMAX'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    station_series = concatenated.sort_values('DATE')['TMAX'].unstack('STATION')
    annual_max = station_series.groupby(level='YEAR').agg('max')
    return annual_max


def process_global_mean_surfacetemp():
    path = os.getenv("GLOBAL_MEAN_TEMP_DATA") or "./data/GLB.Ts+dSST.csv"
    global_mean_temp = pd.read_csv(path, header=1)[['Year', 'J-D']].rename(
        columns={'J-D': 'TEMPANOMALY_GLOB', 'Year': 'YEAR'})
    global_mean_temp.YEAR = global_mean_temp.YEAR.astype(int)
    global_mean_temp.TEMPANOMALY_GLOB = global_mean_temp.TEMPANOMALY_GLOB.astype(float)
    global_mean_temp.YEAR = pd.to_datetime(global_mean_temp.YEAR.astype(str))
    global_mean_temp = global_mean_temp.set_index('YEAR').rolling('1461D').mean()
    global_mean_temp.index = global_mean_temp.index.year
    return global_mean_temp


def get_ghcn_daily_canada_annualmax():
    path_to_directory = os.getenv('CANADA_DATA')
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")] if path_to_directory is not None else [pd.read_csv("./data/british_columbia.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(
        TMAX=lambda x: x['TMAX'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    station_series = concatenated.sort_values('DATE')['TMAX'].unstack('STATION')
    annual_max = station_series.groupby(level='YEAR').agg('max')
    global_mean_temp = process_global_mean_surfacetemp()
    full_dataset = annual_max.merge(global_mean_temp, left_index=True, right_index=True)
    return full_dataset