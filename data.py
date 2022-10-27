import os
from glob import glob
from typing import Sequence

import numpy as np
import pandas as pd
import wget
import xarray as xr

lats = [29, 31]
lons = [-95, -85]
bikaner_coords = [28., 73.3]
jodhpur_coords = [26.3, 73.017]


def download_cpc_us_precip(years: Sequence):
    path_to_directory = os.getenv("LOUISIANA_DATA") + '/gridded/'
    for year in np.arange(*years, step=1):
        if year <= 2006:
            target_url = f"https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/precip.V1.0.{year}.nc"
        else:
            target_url = f"https://downloads.psl.noaa.gov/Datasets/cpc_us_precip/RT/precip.V1.0.{year}.nc"
        if not os.path.isfile(f"{path_to_directory}/precip.V1.0.{year}.nc"):
            try:
                wget.download(target_url, path_to_directory)
                print(f"Downloading url {target_url} to file {path_to_directory}")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def download_ghcn_daily_us(years: Sequence):
    path_to_directory = os.getenv("LOUISIANA_DATA") + '/obs'
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    # cropping to Louisiana region
    stations = stations[(stations.LATITUDE <= lats[-1]) & (stations.LATITUDE >= lats[0]) & (stations.LONGITUDE <= lons[-1]) & (stations.LONGITUDE >= lons[0])]
    for year in np.arange(*years, step=1):
        target_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"
        if not os.path.isfile(f"{path_to_directory}/{year}.csv.gz"):
            try:
                wget.download(target_url, path_to_directory)
                print(f"Downloading url {target_url} to file {path_to_directory}")
                data = pd.read_csv(f"{path_to_directory}/{year}.csv.gz", compression='gzip', header=None, names=["STATION", "DATE", "ELEMENT", "VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "TIME"])
                data = data[(data['ELEMENT'].isin(['PRCP']))].rename(columns={'VALUE': 'PRCP'}).merge(stations, on='STATION').drop(columns=['ELEMENT'])
                data = data.assign(DATE=pd.to_datetime(data.DATE.astype(str)))
                # drop data that have failed one or more quality checks
                data = data[data['Q-FLAG'].isna()].assign(YEAR=data.DATE.dt.year).set_index(['STATION', 'DATE', 'YEAR'])[['PRCP', 'TIME', 'STATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
                data.to_csv(f"{path_to_directory}/{year}.csv")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def download_ghcn_daily_india(years: Sequence):
    path_to_directory = os.getenv("INDIA_DATA")
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    stations = stations[((stations.LATITUDE == bikaner_coords[0]) & (stations.LONGITUDE == bikaner_coords[-1]))
                        | ((stations.LATITUDE == jodhpur_coords[0]) & (stations.LONGITUDE == jodhpur_coords[-1]))]
    for year in np.arange(*years, step=1):
        target_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"
        if not os.path.isfile(f"{path_to_directory}/{year}.csv.gz"):
            try:
                print(f"Downloading url {target_url} to file {path_to_directory}")
                wget.download(target_url, path_to_directory)
                data = pd.read_csv(f"{path_to_directory}/{year}.csv.gz", compression='gzip', header=None, names=["STATION", "DATE", "ELEMENT", "VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "TIME"])
                data = data[(data['ELEMENT'].isin(['TMAX']))].rename(columns={'VALUE': 'TMAX'}).merge(stations, on='STATION').drop(columns=['ELEMENT'])
                data = data.assign(DATE=pd.to_datetime(data.DATE.astype(str)))
                # drop stations for this year when >= 70% unvalid May-June data and drop data with at least 1 quality flag
                data = data[data['Q-FLAG'].isna()].assign(YEAR=data.DATE.dt.year).assign(MONTH=data.DATE.dt.month)
                count_station_year = data[(data['MONTH'] >= 5) & (data['MONTH'] <= 6)].groupby(['STATION']).TMAX.count()
                valid_stations = count_station_year[count_station_year >= 0.7 * 61].index
                data = data.set_index(['STATION', 'DATE', 'YEAR']).loc[valid_stations][['TMAX', 'TIME', 'STATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
                if len(data):
                    data.to_csv(f"{path_to_directory}/{year}.csv")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def get_ghcn_daily_us_annualmax():
    path_to_directory = os.getenv('LOUISIANA_DATA') + '/obs'
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(PREC=lambda x: x['PRCP'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    # filter for stations with at least 10 years (+ of half of the year) of data
    count_station_year = (concatenated.reset_index().groupby('STATION').agg(lambda x: pd.to_timedelta(pd.to_datetime(x.DATE).max() - pd.to_datetime(x.DATE).min())).YEAR).dt.days / 365.25
    valid_stations = count_station_year[count_station_year >= 80].index
    concatenated = concatenated.loc[valid_stations]
    count_daysinyear_station = concatenated.groupby(level=['YEAR', 'STATION']).PREC.count()
    valid_station_year = count_daysinyear_station[count_daysinyear_station >= 365.25 / 2].index
    threedays_mean_prec = concatenated.sort_values('DATE')['PREC'].unstack('STATION').rolling(3, 2).mean()
    annual_max = threedays_mean_prec.groupby(level='YEAR').agg('max').stack().rename('PREC').loc[valid_station_year]
    return annual_max

def get_ghcn_daily_india_annualmax():
    path_to_directory = os.getenv('INDIA_DATA')
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(TMAX=lambda x: x['TMAX'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    station_series = concatenated.sort_values('DATE')['TMAX'].unstack('STATION')
    annual_max = station_series.groupby(level='YEAR').agg('max')
    return annual_max

def get_cpc_us_precip_annualmax_maps():
    path_to_directory = os.getenv("LOUISIANA_DATA") + '/gridded'
    data = xr.open_mfdataset(f"{path_to_directory}/*.nc")
    # Analysis by https://hess.copernicus.org/articles/21/897/2017/hess-21-897-2017.pdf 3-day average precipitation
    local_data = data.sel(lat=slice(lats[0], lats[-1]), lon=slice(lons[0] + 360, lons[-1] + 360))
    threedays_mean_prec = xr.DataArray(np.array(local_data.precip), local_data.precip.coords).rolling(time=3).mean()
    annual_maxima = threedays_mean_prec.resample(time='1Y').max()
    return annual_maxima


def process_global_mean_surfacetemp_for_obs_analysis():
    global_mean_temp = pd.read_csv(os.getenv("LOUISIANA_DATA") + '/GLB.Ts+dSST.csv', header=1)[['Year', 'J-D']].iloc[1:-1].rename(
        columns={'J-D': 'TEMPANOMALY_GLOB', 'Year': 'YEAR'})
    global_mean_temp.TEMPANOMALY_GLOB = global_mean_temp.TEMPANOMALY_GLOB.astype(float)
    global_mean_temp.YEAR = pd.to_datetime(global_mean_temp.YEAR.astype(str))
    global_mean_temp = global_mean_temp.set_index('YEAR').rolling('1461D').mean()
    global_mean_temp.index = global_mean_temp.index.year
    return global_mean_temp


def process_mean_surfacetemp_per_station_for_obs_analysis():
    path_to_directory = os.getenv("LOUISIANA_DATA") + '/obs'
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None, names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    stations = stations[(stations.LATITUDE <= lats[-1]) & (stations.LATITUDE >= lats[0]) & (stations.LONGITUDE <= lons[-1]) & (stations.LONGITUDE >= lons[0])]
    path_to_temp = os.getenv("GLOBAL_MEAN_TEMP_DATA")
    temp = xr.open_mfdataset(path_to_temp).sel(lat=slice(lats[0], lats[-1]), lon=slice(lons[0], lons[-1])).tempanomaly.resample(time='1Y').mean()
    temp_dfs = []
    for s in stations.STATION:
        lat = stations[stations.STATION == s].LATITUDE
        lon = stations[stations.STATION == s].LONGITUDE
        tempanomaly_df = temp.sel(lat=lat, lon=lon, method='nearest', drop=True).to_series() \
            .droplevel(['lat', 'lon']).rename('TEMPANOMALY').reset_index().assign(STATION=s)
        tempanomaly_df = tempanomaly_df.assign(YEAR=lambda x: pd.to_datetime(x.time).dt.year).drop(columns=['time']).set_index(['YEAR', 'STATION'])
        temp_dfs.append(tempanomaly_df)
    temp_df = pd.concat(temp_dfs)
    return temp_df


def process_global_mean_surfacetemp_for_grid_analysis():
    path_to_directory = os.getenv("GLOBAL_MEAN_TEMP_DATA")
    temp = xr.open_mfdataset(path_to_directory).sel(lat=slice(lats[0], lats[-1]), lon=slice(lons[0] + 360, lons[-1] + 360))
    return temp


def build_obs_dataset():
    temp = process_global_mean_surfacetemp_for_obs_analysis()
    prec_annualmax_csv = get_ghcn_daily_us_annualmax()
    full_dataset = prec_annualmax_csv.merge(temp, left_index=True)
    return full_dataset