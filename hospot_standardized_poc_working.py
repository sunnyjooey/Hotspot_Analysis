# Databricks notebook source
!pip install pysal
!pip install descartes
!pip install openpyxl

# COMMAND ----------

# Databricks notebook source
import pandas as pd
import numpy as np
import datetime as dt
import pysal
import geopandas as gpd
import libpysal as lps
from esda.getisord import G_Local

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

# COMMAND ----------

# MAGIC %sh
# MAGIC curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
# MAGIC curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
# MAGIC apt-get update
# MAGIC ACCEPT_EULA=Y apt-get install msodbcsql17
# MAGIC exit

# COMMAND ----------

# read in data
df = pd.read_excel('/dbfs/FileStore/df/undss/sahel_incident_data.xlsx')
df = df[df['Country']=='NIGER']

poly = gpd.read_file('./niger/admin1/NER_adm01_feb2018.shp')

# COMMAND ----------

class HotSpot:
    def __init__(self, df, gdf, df_admin_col, gdf_admin_col):
        self.df = df
        if isinstance(gdf, gpd.geodataframe.GeoDataFrame):
            self.gdf = gdf            
        self.df_admin_col = df_admin_col
        self.gdf_admin_col = gdf_admin_col
    
    def _check_admin(self):
        df_admin_vals = self.df[self.df_admin_col].unique()
        gdf_admin_vals = self.gdf[self.gdf_admin_col].unique()
        bad_vals = [x for x in df_admin_vals if x not in gdf_admin_vals]
        return bad_vals
    
    def correct_df_admin(self, adjust_dict):
        df = self.df
        df[self.df_admin_col] = df[self.df_admin_col].apply(lambda x: adjust_dict[x] if x in adjust_dict.keys() else x)
        df = df[(df[self.df_admin_col] != 'drop') & (~df[self.df_admin_col].isnull())]
        self.df = df
        
    def get_spots_admin(
                        self, 
                        df_col, 
                        sum_count, 
                        weight, 
                        weight_kwargs={}, 
                        glocal_kwargs={'star':True}, 
                        date_filter={}, 
                        seed=8888
                        ):
        
        # check to see that all admin levels are in the shapefile
        bad_vals = self._check_admin()
        if len(bad_vals) > 0:
            raise Exception(f"These admin values in the data are NOT in the geopandas data: {', '.join(bad_vals)}")
        else:
            if len(date_filter) != 0:
                # filter to subset of data by date
                df = self.df
                df.loc[:, date_filter['date_col']] = pd.to_datetime(df[date_filter['date_col']])
                df = df.loc[(df[date_filter['date_col']] >= date_filter['start_date']) & (df[date_filter['date_col']] <= date_filter['end_date']), :]
            else:
                df = self.df
            
            # filter to subset of data by column value
            col = next(iter(df_col))
            val = df_col[col]
            if val is not None:
                df = df.loc[df[col] == val, :]
            
            # sum (like fatalities) or count (where each row is an event) 
            if sum_count == 'sum':
                analysis_df = df.groupby([self.df_admin_col]).agg({col:'sum'}).reset_index()
            elif sum_count == 'count':
                analysis_df = df[[self.df_admin_col, col]].groupby([self.df_admin_col]).count().reset_index()
            else:
                raise Exception("sum_count must be 'sum' or 'count'")
            analysis_col = f'{col}_{sum_count}'
            analysis_df.rename(columns={col:analysis_col}, inplace=True)
            
            # merge with geo dataframe
            fin_gdf = pd.merge(self.gdf[[self.gdf_admin_col]], analysis_df, how='left', left_on=self.gdf_admin_col, right_on=self.df_admin_col)
            fin_gdf.fillna({analysis_col: 0}, inplace=True)
            fin_gdf[analysis_col] = fin_gdf[analysis_col].astype(np.float64)
            fin_gdf = fin_gdf.drop(self.gdf_admin_col, axis=1)
            
            # weights
            wkwargs = {'df': self.gdf, **weight_kwargs}
            if weight == 'q':
                wt = lps.weights.Queen.from_dataframe(**wkwargs)
            elif weight == 'k':
                wt = lps.weights.KNN.from_dataframe(**wkwargs)
            else:
                raise Exception("weight must be 'q' for Queen or 'k' for KNN")
            
            gkwargs = {'y': fin_gdf[analysis_col], 'w': wt, **glocal_kwargs}
            # set for identical results
            np.random.seed(seed)
            G = G_Local(**gkwargs)
            fin_gdf['Gzs'] = G.Zs
            fin_gdf['Gpsim'] = G.p_sim
            
            # save params
            params = {'sum_count':sum_count, 'weight':weight}
            params.update({**df_col, **weight_kwargs, **glocal_kwargs, **date_filter})
            fin_gdf['params'] = [params] * fin_gdf.shape[0]
            
            return fin_gdf

        
    def get_spots_admin_map(
                              self, 
                              df_col, 
                              sum_count, 
                              weight, 
                              weight_kwargs={}, 
                              glocal_kwargs={'star':True}, 
                              date_filter={}, 
                              seed=8888,
                              tresh={'gpsim':0.10, 'gzs':0}
                              ):
        
        # fit
        fin_gdf = self.get_spots_admin(df_col, sum_count, weight, weight_kwargs, glocal_kwargs, date_filter, seed)
        # merge in geo data
        fin_gdf = pd.merge(self.gdf, fin_gdf, how='left', left_on=self.gdf_admin_col, right_on=self.df_admin_col)
        
        conditions = [
                (fin_gdf['Gpsim'] < tresh['gpsim']) & (fin_gdf['Gzs'] > tresh['gzs']),
                (fin_gdf['Gpsim'] < tresh['gpsim']) & (fin_gdf['Gzs'] < tresh['gzs']),
                (fin_gdf['Gpsim'] > tresh['gpsim'])
             ]
        choices = [1, 2, 0]
        fin_gdf['viz'] = np.select(conditions, choices)
        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cold',
                                  markerfacecolor='Blue', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Hot',
                                  markerfacecolor='Red', markersize=10),
                           Line2D([0], [0], marker='o', color='w', label='Not significant',
                                  markerfacecolor='Grey', markersize=10)]

        # Static map
        hmap = colors.ListedColormap(['lightgrey', 'red', 'blue'])
        f, ax = plt.subplots(1, figsize=(9, 9))
        fin_gdf.assign(cl=fin_gdf['viz']).plot(column='cl', categorical=True, k=2, cmap=hmap, linewidth=0.1, ax=ax, edgecolor='white')
        ax.legend(handles=legend_elements, loc='upper right')        
        ax.set_axis_off()
        ax.set_title('boo')
        #plt.savefig(f'/dbfs/FileStore/df/misc/sudan_{m[0].year}_{m[0].month}_{var}.png')
        plt.show()
        

# COMMAND ----------

admin1_map = {'agadez': 'Agadez',
      'zinder': 'Zinder',
      'maradi': 'Maradi',
      'Tllaberi':'Tillabéri',
      'Tillabery':'Tillabéri',
      '0': 'drop'}

date_dict = {'date_col':'Date', 'start_date': dt.datetime(2020,1,1), 'end_date':dt.datetime(2023,2,1)}

# instantiate
hs = HotSpot(df, poly, 'Admin1', 'adm_01')
# correct admin 1 names
hs.correct_df_admin(admin1_map)

# COMMAND ----------

hs_df.iloc[0,-1]

# COMMAND ----------

hs_df = hs.get_spots_admin({'IED':None}, 'sum', 'q', date_filter=date_dict)

# COMMAND ----------

hs.get_spots_admin_map({'IED':None}, 'sum', 'q', date_filter=date_dict)

# COMMAND ----------


