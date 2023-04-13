import os
os.environ['USE_PYGEOS'] = '0'

import pandas as pd
import numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta
import pysal
import geopandas as gpd
import libpysal as lps
from esda.getisord import G_Local
from shapely.geometry import Point

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D


# produce date ranges
def perdelta(start, end, delta):
    lst = []
    curr = start
    while curr < end:
        s = curr
        curr += delta
        e = curr + relativedelta(days=-1)
        lst.append({'start_date':s, 'end_date':e})
    return lst



class HotSpot:
    def __init__(self, gdf, df, gdf_admin_col, df_admin_col=None, df_lat_col=None, df_lon_col=None):
        if isinstance(gdf, gpd.geodataframe.GeoDataFrame):
            self.gdf = gdf  
        else:
            raise Exception("'gdf' must be a geopandas dataframe.")
        self.df = df
        self.gdf_admin_col = gdf_admin_col
        self.df_admin_col = df_admin_col
        self.df_lat_col = df_lat_col
        self.df_lon_col = df_lon_col
        self.processed_df = None
        self.process_params = None
    
    def _check_admin(self):
        df_admin_vals = self.df[self.df_admin_col].unique()
        gdf_admin_vals = self.gdf[self.gdf_admin_col].unique()
        bad_vals = [x for x in df_admin_vals if x not in gdf_admin_vals]
        bad_vals.sort()
        return bad_vals
    
    def correct_df_admin(self, adjust_dict):
        df = self.df.copy()
        df[self.df_admin_col] = df[self.df_admin_col].apply(lambda x: adjust_dict[x] if x in adjust_dict.keys() else x)
        df = df[(df[self.df_admin_col] != 'drop') & (~df[self.df_admin_col].isnull())]
        self.df = df
    
    def process_df(self, target_dict, filter_dict={}, date_dict={}, geo='admin'):
        ##### check
        # check that we have admin column or lat/lon columns
        if geo == 'admin':
            if self.df_admin_col is None:
                raise Exception("You must provide 'df_admin_col'")
            else:
                bad_vals = self._check_admin()
                if len(bad_vals) > 0:
                    raise Exception(f"These admin values in the data are NOT in the geopandas data: {', '.join(bad_vals)}")
        elif geo == 'coord':
            if (self.df_lat_col is None) or (self.df_lon_col is None):
                raise Exception("You must provide both 'df_lat_col' and 'df_lon_col'")
        else:
            raise Exception("'geo' can be 'admin' or 'coord'")
            
        ##### filter
        # filter to subset of data by date
        df = self.df.copy()
        if len(date_dict) > 0:
            df = df.loc[(df[date_dict['date_col']] >= date_dict['start_date']) & (df[date_dict['date_col']] <= date_dict['end_date']), :]        
            
        # filter to subset of data by column values
        for col, val_lst in filter_dict.items():
            df = df.loc[df[col].isin(val_lst), :]      
        
        ##### geo matching type
        # if going by lat/lon columns
        if geo == 'coord':
            # Define geometry of events data
            geometry = [Point(xy)  for xy in zip(df[self.df_lon_col], df[self.df_lat_col])]
            # Build spatial data frame
            df = gpd.GeoDataFrame(df, crs=self.gdf.crs, geometry=geometry)
            # Create merged spatial data frame to confirm matching dimensions
            df = gpd.sjoin(self.gdf, df, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')
            df = df[[self.gdf_admin_col, col]]
        # if going by admin level names - this is to make uniform column names
        else:
            df.rename(columns={self.df_admin_col: self.gdf_admin_col}, inplace=True)

        ##### agg
        # sum/count: sum (like fatalities) or count (where each row is an event) 
        df_col = target_dict['tgt_col']
        sum_count = target_dict['agg_typ']
        if sum_count == 'sum':
            analysis_df = df.groupby([self.gdf_admin_col]).agg({df_col:'sum'}).reset_index()
        elif sum_count == 'count':
            analysis_df = df[[self.gdf_admin_col, df_col]].groupby([self.gdf_admin_col]).count().reset_index()
        else:
            raise Exception("'agg_typ' in 'target_dict' must be 'sum' or 'count'")
        analysis_df.rename(columns={df_col:'num'}, inplace=True)
        
        ##### save
        tm = dt.datetime.now().strftime("%Y%m%d%H%M%S")
        idx = f'{df_col}_{sum_count}_{tm}'
        target_dict.update({'id': idx})
        target_dict.update(filter_dict)
        target_dict.update(date_dict)
        target_dict.pop('date_col', None)
        
        ##### set attribute
        self.processed_df = analysis_df
        self.process_params = target_dict
        
    def get_spots_df(
            self, 
            weight, 
            weight_kwargs={}, 
            glocal_kwargs={'star':True}, 
            seed=8888,
            get_map=True,
            tresh={'gpsim':0.10, 'gzs':0}):

        if self.processed_df is None:
            raise Exception("'process_df' first!")
        else:
            analysis_df = self.processed_df.copy()
            
        # merge with geo dataframe
        fin_gdf = pd.merge(self.gdf[[self.gdf_admin_col]], analysis_df, how='left', left_on=self.gdf_admin_col, right_on=self.gdf_admin_col)
        fin_gdf.fillna({'num': 0}, inplace=True)
        fin_gdf['num'] = fin_gdf['num'].astype(np.float64)
        
        # weights
        wkwargs = {'df': self.gdf, **weight_kwargs}
        if weight == 'q':
            wt = lps.weights.Queen.from_dataframe(**wkwargs)
        elif weight == 'k':
            wt = lps.weights.KNN.from_dataframe(**wkwargs)
        else:
            raise Exception("weight must be 'q' for Queen or 'k' for KNN")
        
        # fit
        gkwargs = {'y': fin_gdf['num'], 'w': wt, **glocal_kwargs}
        # set for identical results
        np.random.seed(seed)
        G = G_Local(**gkwargs)
        fin_gdf['Gzs'] = G.Zs
        fin_gdf['Gpsim'] = G.p_sim

        # save params
        params = {'weight':weight}
        params.update({**weight_kwargs, **glocal_kwargs})
        fin_gdf['process_params'] = [self.process_params] * fin_gdf.shape[0]
        fin_gdf['model_params'] = [params] * fin_gdf.shape[0]

        # map
        if get_map:
            # merge in geo data
            fin_gdf1 = pd.merge(self.gdf, fin_gdf, how='left', left_on=self.gdf_admin_col, right_on=self.gdf_admin_col)
            
            # condition / thresholds for map viz
            conditions = [
                    (fin_gdf1['Gpsim'] < tresh['gpsim']) & (fin_gdf1['Gzs'] > tresh['gzs']),
                    (fin_gdf1['Gpsim'] < tresh['gpsim']) & (fin_gdf1['Gzs'] < tresh['gzs']),
                    (fin_gdf1['Gpsim'] > tresh['gpsim'])
                ]
            choices = [1, 2, 0]
            fin_gdf1['viz'] = np.select(conditions, choices)
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cold',
                                    markerfacecolor='Blue', markersize=10),
                            Line2D([0], [0], marker='o', color='w', label='Hot',
                                    markerfacecolor='Red', markersize=10),
                            Line2D([0], [0], marker='o', color='w', label='Not significant',
                                    markerfacecolor='Grey', markersize=10)]
            # Title
            fil = fin_gdf1.loc[0, 'process_params']
            par = fin_gdf1.loc[0, 'model_params']

            try:
                date = f"- {fil['start_date'].year}/{fil['start_date'].month}/{fil['start_date'].day} to {fil['end_date'].year}/{fil['end_date'].month}/{fil['end_date'].day}"
                title = f"{fil['tgt_col']} {fil['agg_typ']} - weight {par['weight']} {date}"
            except:
                title = ''
            
            
            # Static map
            hmap = colors.ListedColormap(['lightgrey', 'red', 'blue'])
            f, ax = plt.subplots(1, figsize=(9, 9))
            fin_gdf1.assign(cl=fin_gdf1['viz']).plot(column='cl', categorical=True, k=2, cmap=hmap, linewidth=0.1, ax=ax, edgecolor='white')
            ax.legend(handles=legend_elements, loc='upper right')        
            ax.set_axis_off()
            ax.set_title(title)
            # plt.savefig(f'/dbfs/FileStore/df/misc/sudan_{m[0].year}_{m[0].month}_{var}.png')
            plt.show()

        return fin_gdf

