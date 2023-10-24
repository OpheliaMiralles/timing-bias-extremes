import os
from datetime import date
from functools import partial
from glob import glob
from pathlib import Path
from typing import Sequence

import cdsapi as cdsapi
import geopandas
import numpy as np
import pandas as pd
import shapely
import wget
import xarray as xr
from shapely.geometry import box

bikaner_coords = [28., 73.3]
jodhpur_coords = [26.3, 73.017]
seattle = [47.4444, -122.3139]
portland = [45.5908, -122.6003]


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


def download_ghcn_daily_portland_seattle(years: Sequence):
    path_to_directory = os.getenv("CANADA_DATA")
    stations = pd.read_table(f"{path_to_directory}/ghcnd-stations.txt", sep='\s+', usecols=[0, 1, 2, 3, 4, 5], header=None,
                             names=['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'STATE', 'NAME'])
    stations = stations[((stations.LATITUDE == seattle[0]) & (stations.LONGITUDE == seattle[-1]))
                        | ((stations.LATITUDE == portland[0]) & (stations.LONGITUDE == portland[-1]))]
    for year in np.arange(*years, step=1):
        target_url = f"https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/{year}.csv.gz"
        if not os.path.isfile(f"{path_to_directory}/{year}.csv.gz"):
            try:
                print(f"Downloading url {target_url} to file {path_to_directory}")
                wget.download(target_url, path_to_directory)
                data = pd.read_csv(f"{path_to_directory}/{year}.csv.gz", compression='gzip', header=None, names=["STATION", "DATE", "ELEMENT", "VALUE", "M-FLAG", "Q-FLAG", "S-FLAG", "TIME"])
                data = data[(data['ELEMENT'].isin(['TMAX']))].rename(columns={'VALUE': 'TMAX'}).merge(stations, on='STATION').drop(columns=['ELEMENT'])
                data = data.assign(DATE=pd.to_datetime(data.DATE.astype(str)))
                data = data[data['Q-FLAG'].isna()].assign(YEAR=data.DATE.dt.year).assign(MONTH=data.DATE.dt.month)
                data = data.set_index(['STATION', 'DATE', 'YEAR'])[['TMAX', 'TIME', 'STATE', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION']]
                if len(data):
                    data.to_csv(f"{path_to_directory}/{year}.csv")
            except Exception as err:
                print(f"---> Can't access {target_url}: {err}")
        else:
            print(f"Year {year} has already been downloaded to {path_to_directory}")


def get_ghcn_daily_india_annualmax():
    path_to_directory = os.getenv('INDIA_DATA')
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(TMAX=lambda x: x['TMAX'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    station_series = concatenated.sort_values('DATE')['TMAX'].unstack('STATION')
    annual_max = station_series.groupby(level='YEAR').agg('max')
    return annual_max


def process_global_mean_surfacetemp():
    global_mean_temp = pd.read_csv(os.getenv("GLOBAL_MEAN_TEMP_DATA"), header=1)[['Year', 'J-D']].rename(
        columns={'J-D': 'TEMPANOMALY_GLOB', 'Year': 'YEAR'})
    global_mean_temp.YEAR = global_mean_temp.YEAR.astype(int)
    global_mean_temp.TEMPANOMALY_GLOB = global_mean_temp.TEMPANOMALY_GLOB.astype(float)
    global_mean_temp.YEAR = pd.to_datetime(global_mean_temp.YEAR.astype(str))
    global_mean_temp = global_mean_temp.set_index('YEAR').rolling('1461D').mean()
    global_mean_temp.index = global_mean_temp.index.year
    return global_mean_temp


def get_ghcn_daily_canada_annualmax():
    path_to_directory = os.getenv('CANADA_DATA')
    csvs = [pd.read_csv(f) for f in glob(f"{path_to_directory}/*.csv")]
    csvs = [csv.assign(YEAR=pd.to_datetime(csv.DATE).dt.year).set_index(['STATION', 'DATE', 'YEAR']).assign(TMAX=lambda x: x['TMAX'] / 10) for csv in csvs]
    concatenated = pd.concat(csvs)
    station_series = concatenated.sort_values('DATE')['TMAX'].unstack('STATION')
    annual_max = station_series.groupby(level='YEAR').agg('max')
    global_mean_temp = process_global_mean_surfacetemp()
    full_dataset = annual_max.merge(global_mean_temp, left_index=True, right_index=True)
    return full_dataset

# DATA FOR GLOBAL HEAT 2023
def download_ERA5_temperature(datapath: Path, file_suffix: str, start_date, end_date):
    c = cdsapi.Client()
    request_args = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'time': [
            '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
        ],
        'variable': ['2m_temperature']
    }
    for date in pd.date_range(start_date, end_date):
        filename = f'{date.year}{date.month:02}{date.day:02}_{file_suffix}'
        dest = Path(datapath).joinpath(filename).with_suffix('.nc')
        if dest.exists():
            print(f"File {filename} already exists")
            try:
                ds = xr.open_dataset(dest)
                max_temp = ds.assign_coords({'date': [ds.time[0].values]}).max(['time'])
                max_temp.to_netcdf(dest)
            except:
                print(f"File {filename} already processed")
        else:
            dest.parent.mkdir(exist_ok=True)
            date_request = {**request_args, 'date': date.strftime('%Y-%m-%d')}
            c.retrieve('reanalysis-era5-single-levels', date_request, str(dest))
            ds = xr.open_dataset(dest)
            max_temp = ds.assign_coords({'date': [ds.time[0].values]}).max(['time'])
            max_temp.to_netcdf(dest)


def download_ERA5(datapath, start_date=date(2001, 12, 22), end_date=date(2023, 8, 1)):    download_ERA5_temperature(datapath, 'era5_daily_temp', start_date, end_date)def _preprocess_gpd(x, polygon_gpd):    y = x.assign_coords({'longitude': xr.where(x.longitude <= 180, x.longitude, x.longitude - 360)}).sortby('longitude')    ry = y.t2m.rio.set_crs('epsg:4326')    y_clipped = ry.rio.clip(polygon_gpd.geometry)    ds_clipped_degrees = y_clipped.to_dataset().drop('spatial_ref') - 273.15    return ds_clipped_degrees.assign_coords({'date': y.date}).mean(dim=['longitude', 'latitude'])def get_us_mex_polygon():    path_to_shapefile = './data/regions/us_west.shp'    if os.path.isfile(path_to_shapefile):        return geopandas.read_file(path_to_shapefile).to_crs(epsg=4326)    path_to_us_states = './data/gz_2010_us_040_00_500k/gz_2010_us_040_00_500k.shp'    states = geopandas.read_file(path_to_us_states)    valid_us_states = ['California', 'Nevada', 'Utah', 'Arizona', 'New Mexico', 'Texas']    polygon_us = states[states.NAME.isin(valid_us_states)][['NAME', 'geometry']].to_crs(epsg=4326).set_index('NAME')    # Mexico    path_to_mex_states = './data/mex_admbnda_govmex_20210618_SHP/mex_admbnda_adm1_govmex_20210618.shp'    mex_states = geopandas.read_file(path_to_mex_states)    valid_mex_states = ['Baja California', 'Baja California Sur', 'Chihuahua', 'Sonora', 'Coahuila de Zaragoza', 'Tamaulipas', 'Nuevo León']    polygon_mex = mex_states[mex_states.ADM1_ES.isin(valid_mex_states)].rename(columns={'ADM1_ES': 'NAME'})[['NAME', 'geometry']].set_index('NAME')    polygon = geopandas.overlay(polygon_us, polygon_mex, how='union')    return polygondef get_eastern_us_polygon():    path_to_shapefile = './data/regions/us_east.shp'    if os.path.isfile(path_to_shapefile):        return geopandas.read_file(path_to_shapefile).to_crs(epsg=4326)    path_to_us_states = './data/gz_2010_us_040_00_500k/gz_2010_us_040_00_500k.shp'    lats = [60, 40]    lons = [-95, -30]    s = box(lons[0], lats[-1], lons[-1], lats[0])    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    bb = glob_countries.intersection(s).to_frame('geometry')    arctic = get_arctic_polygon()    polygon_gpd = geopandas.overlay(bb[bb.area > 0], arctic, 'difference')    states = geopandas.read_file(path_to_us_states)    valid_us_states = ['Minnesota', 'Wisconsin', 'Michigan', 'New York', 'Connecticut', 'Maryland', 'New Jersey', 'Delaware', 'Vermont', 'Rhode Island',                       'Massachusetts', 'New Hampshire',                       'Pennsylvania', 'Ohio', 'Indiana', 'Maine', 'Kentucky', 'West Virginia', 'Virginia',                       'Tennessee', 'North Carolina', 'South Carolina', 'Mississippi', 'Alabama',                       'Georgia', 'Florida', 'Illinois']    polygon_us = states[states.NAME.isin(valid_us_states)][['NAME', 'geometry']].to_crs(epsg=4326).set_index('NAME')    polygon = geopandas.overlay(polygon_us, polygon_gpd, 'union')    return polygondef get_central_us_polygon():    path_to_shapefile = './data/regions/us_center.shp'    if os.path.isfile(path_to_shapefile):        return geopandas.read_file(path_to_shapefile).to_crs(epsg=4326)    path_to_us_states = './data/gz_2010_us_040_00_500k/gz_2010_us_040_00_500k.shp'    lats = [80, 40]    lons = [-180, -90]    s = box(lons[0], lats[-1], lons[-1], lats[0])    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    bb = glob_countries.intersection(s).to_frame('geometry')    polygon_gpd = geopandas.overlay(bb[bb.area > 0], get_arctic_polygon(), 'difference')    states = geopandas.read_file(path_to_us_states)    invalid_us_states = ['Minnesota', 'Wisconsin', 'Michigan', 'New York', 'Connecticut', 'Maryland', 'New Jersey', 'Delaware', 'Vermont', 'Rhode Island',                         'Pennsylvania', 'Ohio', 'Maine', 'Kentucky', 'West Virginia', 'Virginia', 'Massachusetts', 'New Hampshire',                         'Tennessee', 'North Carolina', 'South Carolina', 'Mississippi', 'Alabama',                         'Georgia', 'Florida', 'Illinois', 'Indiana', 'California', 'Nevada', 'Utah', 'Arizona', 'New Mexico', 'Texas']    polygon_us = states[~states.NAME.isin(invalid_us_states)][['NAME', 'geometry']].to_crs(epsg=4326).set_index('NAME')    arctic = get_arctic_polygon()    polygon_us = geopandas.overlay(polygon_us, arctic, 'difference')    polygon = geopandas.overlay(polygon_us, polygon_gpd, 'union')    return polygondef get_southern_europe_polygon():    path_to_south_europe_shapefile = './data/regions/southern_europe.shp'    if os.path.isfile(path_to_south_europe_shapefile):        return geopandas.read_file(path_to_south_europe_shapefile).to_crs(epsg=4326)    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    lats_europe = [45, 22.2]    lons_europe = [-25, 36.2]    s = box(lons_europe[0], lats_europe[-1], lons_europe[-1], lats_europe[0])    polygon_gpd = glob_countries.intersection(s)    return polygon_gpd[polygon_gpd.area > 0].to_frame('geometry')def get_northern_europe_polygon():    path_to_north_europe_shapefile = './data/regions/northern_europe.shp'    if os.path.isfile(path_to_north_europe_shapefile):        return geopandas.read_file(path_to_north_europe_shapefile).to_crs(epsg=4326)    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    lats_europe = [80, 45]    lons_europe = [-10, 60]    s = box(lons_europe[0], lats_europe[-1], lons_europe[-1], lats_europe[0])    polygon_gpd = glob_countries.intersection(s)    arctic = get_arctic_polygon()    polygon_gpd = geopandas.overlay(polygon_gpd.to_frame('geometry'), arctic, 'difference')    return polygon_gpddef get_eastern_asia_polygon():    path_to_eastern_asia_shapefile = './data/regions/eastern_asia.shp'    if os.path.isfile(path_to_eastern_asia_shapefile):        return geopandas.read_file(path_to_eastern_asia_shapefile).to_crs(epsg=4326)    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    lats_asia = [80, 45]    lons_asia = [60, 180]    s = box(lons_asia[0], lats_asia[-1], lons_asia[-1], lats_asia[0])    polygon_gpd = glob_countries.intersection(s)    arctic = get_arctic_polygon()    china = get_china_polygon()    polygon_gpd = geopandas.overlay(polygon_gpd.to_frame('geometry'), arctic, 'difference')    polygon_gpd = geopandas.overlay(polygon_gpd, china, 'difference')    return polygon_gpddef get_china_polygon():    path_to_china_shapefile = './data/regions/china.shp'    if os.path.isfile(path_to_china_shapefile):        return geopandas.read_file(path_to_china_shapefile).to_crs(epsg=4326)    path_to_china_states = './data/china_provinces/chn_adm_ocha_2020_shp/chn_admbnda_adm1_ocha_2020.shp'    states = geopandas.read_file(path_to_china_states).set_crs(epsg=4326)    valid_states = ['Hebei Province', 'Shanxi Province', 'Shaanxi Province', 'Hong Kong Special Administrative Region', 'Inner Mongolia Autonomous Region',                    'Ningxia Hui Autonomous Region', 'Gansu province', 'Hubei Province', 'Hunan Province', 'Henan Province', 'Jilin Province', 'Taiwan Province',                    'Guangxi Zhuang Autonomous Region', 'Yunnan Province', 'Guangdong Province', 'Guizhou Province', 'Macao Special Administrative Region',                    'Shandong Province', 'Heilongjiang Province', 'Zhejiang Province', 'Jiangsu Province', 'Jiangxi Province', 'Fujian Province', 'Liaoning Province',                    'Anhui Province', 'Beijing Municipality', 'Shanghai Municipality', 'Tianjin Municipality', 'Chongqing Municipality', 'Hainan Province']    polygon = states[states.ADM1_EN.isin(valid_states)].rename(columns={'ADM1_EN': 'NAME'})[['NAME', 'geometry']].set_index('NAME')    return polygondef get_middle_east_polygon():    china = get_china_polygon()    eu_north = get_northern_europe_polygon()    asia = get_eastern_asia_polygon()    path_to_middle_east_shapefile = './data/regions/middle_east.shp'    if os.path.isfile(path_to_middle_east_shapefile):        return geopandas.read_file(path_to_middle_east_shapefile).to_crs(epsg=4326)    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    lats_me = [50, 22.2]    lons_me = [36.2, 115]    s = box(lons_me[0], lats_me[-1], lons_me[-1], lats_me[0])    polygon_gpd = glob_countries.intersection(s)    polygon_gpd = geopandas.overlay(polygon_gpd.to_frame('geometry'), china, 'difference')    polygon_gpd = geopandas.overlay(polygon_gpd, eu_north, 'difference')    polygon_gpd = geopandas.overlay(polygon_gpd, asia, 'difference')    return polygon_gpddef get_se_asia_polygon():    china = get_china_polygon()    eastern_asia = get_eastern_asia_polygon()    me = get_middle_east_polygon()    path_to_se_asia_shapefile = './data/regions/se_asia.shp'    if os.path.isfile(path_to_se_asia_shapefile):        return geopandas.read_file(path_to_se_asia_shapefile).to_crs(epsg=4326)    path_to_europe_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_europe_shapefile).to_crs(epsg=4326)    lats_me = [50, 22.2]    lons_me = [115, 180]    s = box(lons_me[0], lats_me[-1], lons_me[-1], lats_me[0])    polygon_gpd = glob_countries.intersection(s)    polygon_gpd = geopandas.overlay(polygon_gpd.to_frame('geometry'), china, 'union')    polygon_gpd = geopandas.overlay(polygon_gpd, eastern_asia, 'difference')    polygon_gpd = geopandas.overlay(polygon_gpd, me, 'difference')    return polygon_gpddef get_arctic_polygon():    path_to_arctic_shapefile = './data/regions/arctic.shp'    if os.path.isfile(path_to_arctic_shapefile):        return geopandas.read_file(path_to_arctic_shapefile).to_crs(epsg=4326)    path_to_countries_shapefile = './data/ref-countries-2020-01m.shp/CNTR_RG_01M_2020_3035.shp/CNTR_RG_01M_2020_3035.shp'    glob_countries = geopandas.read_file(path_to_countries_shapefile).to_crs(epsg=4326)    arctic = geopandas.read_file('./data/arctic_zones_complete_polygons_4326/arctic_zones_complete_polygons_4326.shp')    arctic_flipped = geopandas.GeoSeries(arctic['geometry']).map(lambda polygon: shapely.ops.transform(lambda x, y: (y, x), polygon)).to_frame('geometry').set_crs('epsg:4326')    arctic_countries = geopandas.overlay(glob_countries, arctic_flipped, how='intersection')    return arctic_countriesdef get_regional_txn(polygon, n, region_name):    path_to_directory = os.getenv("GLOBAL_ERA5_TEMP_DATA")    partial_func = partial(_preprocess_gpd, polygon_gpd=polygon)    to_concat = []    for year in np.arange(1950, 2024, 1):        print(f'Year {year} for {region_name}')        ds = xr.open_mfdataset(            sorted(glob(f'{path_to_directory}/{year}*.nc')), preprocess=partial_func        ).rolling(dim={'date': n}).mean().assign(year=lambda x: x.date.dt.year).groupby('year').max()        to_concat.append(ds.t2m.to_dataframe(name=region_name))    csv = pd.concat(to_concat)    return csvdef save_european_temp():    csv = get_regional_txn(get_arctic_polygon(), 7, 'Arctic')    csv.to_csv('./data/era5_arctic.csv')    csv = get_regional_txn(get_southern_europe_polygon(), 7, 'Southern Europe')    csv.to_csv('./data/era5_south_europe.csv')    csv = get_regional_txn(get_northern_europe_polygon(), 7, 'Northern Europe')    csv.to_csv('./data/era5_north_europe.csv')def save_us_temp():    csv = get_regional_txn(get_us_mex_polygon(), 18, 'USA/Mexico')    csv.to_csv('./data/era5_us_west.csv')    csv = get_regional_txn(get_central_us_polygon(), 18, 'Central USA')    csv.to_csv('./data/era5_us_center.csv')    csv = get_regional_txn(get_eastern_us_polygon(), 18, 'East Coast USA')    csv.to_csv('./data/era5_us_east.csv')def save_asia_temp():    csv = get_regional_txn(get_eastern_asia_polygon(), 7, 'North-Eastern Asia')    csv.to_csv('./data/era5_me_asia.csv')    csv = get_regional_txn(get_middle_east_polygon(), 14, 'Middle East')    csv.to_csv('./data/era5_me_asia.csv')    csv = get_regional_txn(get_se_asia_polygon(), 14, 'South-Eastern Asia')    csv.to_csv('./data/era5_se_asia.csv')