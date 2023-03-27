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

# MAGIC %sh
# MAGIC curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
# MAGIC curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
# MAGIC apt-get update
# MAGIC ACCEPT_EULA=Y apt-get install msodbcsql17
# MAGIC exit

# COMMAND ----------

from keys import keys

# ACLED data
database_host = keys["database_host"]
database_port = keys["database_port"]
database_name = keys["database_name"]
user = keys["user"]
password = keys["password"]

table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

df1 = (spark.read
  .format("com.microsoft.sqlserver.jdbc.spark")
  .option("url", url)
  .option("dbtable", table)
  .option("user", user)
  .option("password", password)
  .load()
)

df1 = df1.filter(df1.CountryFK==201)
df1 = df1.toPandas()

# Convert ACLED Dates to pd
def convert_dt(value):
    valstr = str(value)
    date_clean = dt.datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
    return date_clean

df1.loc[:, 'TimeFK_Event_Date'] = df1['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))

# COMMAND ----------

# undss data
df2 = pd.read_excel('/dbfs/FileStore/df/undss/sahel_incident_data.xlsx')
df2 = df2[df2['Country']=='NIGER']

# change date column to datetime
df2.loc[:, 'Date'] = pd.to_datetime(df2['Date'])

# COMMAND ----------

# shapefile niger
poly = gpd.read_file('./niger/admin2/NER_adm02_feb2018.shp')

# COMMAND ----------

# dict of date filter
date_filter = {'date_col':'Date', 'start_date': dt.datetime(2022,8,1), 'end_date':dt.datetime(2023,1,31)}

# COMMAND ----------

# instantiate
hs = HotSpot(poly, df2, gdf_admin_col='adm_02', df_admin_col='Admin2')
# filter/process - will not work
hs.process_df({'df_col':'VBIED'}, 'sum', date_filter, 'admin')

# COMMAND ----------

# correct admin 1 names
admin1_map = {'agadez': 'Agadez',
      'zinder': 'Zinder',
      'maradi': 'Maradi',
      'Tllaberi':'Tillabéri',
      'Tillabery':'Tillabéri',
      '0': 'drop'}

hs.correct_df_admin(admin1_map)

# COMMAND ----------

# now will work
hs.process_df({'df_col':'IED'}, 'sum', date_filter, 'admin')
hs.processed_df

# COMMAND ----------

hs_df = hs.get_spots_df('q')
hs_df

# COMMAND ----------

hs.get_spots_map('q')

# COMMAND ----------

# instantiate / process
hs = HotSpot(poly, df1, 'adm_01', None, 'ACLED_Latitude', 'ACLED_Longitude')
hs.process_df({'df_col':'ACLED_Event_Type', 'col_val':'Protests'}, 'count', {}, 'coord')

# COMMAND ----------

# map
hs.get_spots_map('q')

# COMMAND ----------


