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

# COMMAND ----------

# undss data
df2 = pd.read_excel('/dbfs/FileStore/df/undss/sahel_incident_data.xlsx')
df2 = df2[df2['Country']=='NIGER']

# COMMAND ----------

# shapefile niger
poly = gpd.read_file('./niger/admin1/NER_adm01_feb2018.shp')

# COMMAND ----------

date_dict = {'date_col':'Date', 'start_date': dt.datetime(2020,1,1), 'end_date':dt.datetime(2023,2,1)}

# instantiate
hs = HotSpot(poly, df2, 'adm_01', 'Admin1')

# COMMAND ----------

# will not work
hs_df = hs.get_spots_df({'df_col':'IED'}, 'sum', 'q', date_filter=date_dict)

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
hs_df = hs.get_spots_df({'df_col':'IED'}, 'sum', 'q', date_filter=date_dict)
hs_df

# COMMAND ----------

hs.get_spots_map({'df_col':'IED'}, 'sum', 'q', date_filter=date_dict)

# COMMAND ----------

# instantiate
hs = HotSpot(poly, df1, 'adm_01', None, 'ACLED_Latitude', 'ACLED_Longitude')
hs_df = hs.get_spots_df({'df_col':'ACLED_Event_Type', 'col_val':'Protests'}, 'count', 'q')
hs_df

# COMMAND ----------

hs.get_spots_map({'df_col':'ACLED_Event_Type', 'col_val':'Protests'}, 'count', 'q')

# COMMAND ----------


