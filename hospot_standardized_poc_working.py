# Databricks notebook source
!pip install pysal
!pip install descartes
!pip install openpyxl

# COMMAND ----------

import pandas as pd
import datetime as dt
import geopandas as gpd
from hotspot import HotSpot

# COMMAND ----------

# bangladesh data
df2 = pd.read_csv('/dbfs/FileStore/df/bangladesh/all_political.csv')
# change date column to datetime
df2.loc[:, 'eventdate'] = pd.to_datetime(df2['eventdate'])
df2.head(3)

# COMMAND ----------

# shapefile niger
poly = gpd.read_file('./bangladesh/admin1/bgd_admbnda_adm1_bbs_20201113.shp')

# COMMAND ----------

# dict of date filter
date_filter = {'date_col':'eventdate', 'start_date': dt.datetime(2022,1,1), 'end_date':dt.datetime(2022,12,31)}

# COMMAND ----------

# instantiate
hs = HotSpot(poly, df2, gdf_admin_col='ADM1_EN', df_admin_col='division')

# COMMAND ----------

# filter/process - will not work
hs.process_df({'tgt_col':'mtvincidentone', 'agg_typ':'count'}, {'mtvincidentone':['Elections']}, date_filter, 'admin')

# COMMAND ----------

# correct admin 1 names
admin1_map = {'Barishal': 'Barisal',
      'Chattogram': 'Chittagong'}

hs.correct_df_admin(admin1_map)

# COMMAND ----------

# now will work
hs.process_df({'tgt_col':'mtvincidentone', 'agg_typ':'count'}, {'mtvincidentone':['Elections']}, date_filter, 'admin')
hs.processed_df

# COMMAND ----------

# get hot spots - queen method
hs_df = hs.get_spots_df('q')
hs_df

# COMMAND ----------

# produce date ranges
from dateutil.relativedelta import relativedelta
def perdelta(start, end, delta):
    lst = []
    curr = start
    while curr < end:
        s = curr
        curr += delta
        e = curr + relativedelta(days=-1)
        lst.append({'start_date':s, 'end_date':e})
    return lst

dlst = perdelta(dt.datetime(2012, 1, 1), dt.datetime(2022, 12, 31), relativedelta(years=1))

# COMMAND ----------

# dict of date filter
for date_filter in dlst:
    date_filter.update({'date_col': 'eventdate'})
    hs.process_df({'tgt_col':'mtvincidentone', 'agg_typ':'count'}, {'mtvincidentone':['Elections']}, date_filter, 'admin')
    hs.get_spots_map('q')

# COMMAND ----------


