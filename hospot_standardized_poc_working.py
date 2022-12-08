# Databricks notebook source
!pip install pysal
!pip install descartes

# COMMAND ----------

# Databricks notebook source
import pandas as pd
import numpy as np
import time
from itertools import count
import pysal
import esda
import geopandas as gpd
from geopandas import GeoDataFrame
import libpysal as lps
# import matplotlib.pyplot as plt
# from pysal.mapclassify import K
# from seaborn.palettes import color_palette
from shapely.geometry import Point
# import openpyxl
from esda.getisord import G_Local
import datetime as dt

# COMMAND ----------

# MAGIC %sh
# MAGIC curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
# MAGIC curl https://packages.microsoft.com/config/ubuntu/16.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
# MAGIC apt-get update
# MAGIC ACCEPT_EULA=Y apt-get install msodbcsql17
# MAGIC exit

# COMMAND ----------

from keys import keys

database_host = keys["database_host"]
database_port = keys["database_port"]
database_name = keys["database_name"]
user = keys["user"]
password = keys["password"]

table = "dbo.CRD_ACLED"
url = f"jdbc:sqlserver://{database_host}:{database_port};databaseName={database_name};"

df = (spark.read
  .format("com.microsoft.sqlserver.jdbc.spark")
  .option("url", url)
  .option("dbtable", table)
  .option("user", user)
  .option("password", password)
  .load()
)

df = df.filter((df.CountryFK==214) & (df.ACLED_Year=='2022')) 

# COMMAND ----------

#Set for identical results
np.random.seed(2021)
#Import Relevant Country Shapefile
poly = gpd.read_file('adm2/SDN_adm2.shp')

# COMMAND ----------

# Filt data  if of interest
# def convert_dt(value):
#     valstr = str(value)
#     date_clean = dt.datetime(year=int(valstr[0:4]), month=int(valstr[4:6]), day=int(valstr[6:8]))
#     return date_clean

# conflict['TimeFK_Event_Date'] = conflict['TimeFK_Event_Date'].apply(lambda x: convert_dt(x))

# conflict = conflict[(conflict['TimeFK_Event_Date'] > '20210801') & (conflict['TimeFK_Event_Date'] < '20211231')]

conflict = df.toPandas()

# COMMAND ----------

#Define geometry of events data
geometry = [Point(xy)  for xy in zip(conflict['ACLED_Longitude'], conflict['ACLED_Latitude'])]
crs = 'epsg:4326'

#Build spatial data frame
conflict_geo = GeoDataFrame(conflict, crs=crs, geometry=geometry)

# COMMAND ----------

#Create merged spatial data frame to confirm matching dimensions
sj_gdf = gpd.sjoin(poly, conflict_geo, how='inner', predicate='intersects', lsuffix='left', rsuffix='right')

list(sj_gdf)

# COMMAND ----------

sj_gdf.head()

# COMMAND ----------

#############
# Generrate counts variables
#############
#Fatalities
Total_f_gdf = sj_gdf['ACLED_Fatalities'].groupby([sj_gdf['NAME_2']]).sum()

#Total Events
Total_e_gdf = sj_gdf['NAME_2'].groupby([sj_gdf['NAME_2']]).count()
Total_e_gdf.rename('Event Count', inplace=True)

####Create event type df
#protests
prot = sj_gdf.loc[sj_gdf['ACLED_Event_Type'] == "Protests"].groupby(['NAME_2']).agg({'ACLED_Event_Type':'count'}).squeeze()
prot.rename('Protest Count', inplace=True)


#######Actor Type - Did not do this yet (mainly as ACLED['Actor_Type'].value_counts() in the sri lanka case was not promising) - maybe look into interaction

####Concatenate dataframes
merged_df = pd.concat([Total_e_gdf, Total_f_gdf, prot],axis=1)

# COMMAND ----------

#Merge with geospatial dataframe
fin_gdf = poly.join(merged_df, on='NAME_2')
#fin_gdf = fin_gdf.join(Total_e_gdf, on='NA')

#Assumption here for ACLED is that if there is no event of that type in a polygon then none have happened
# fin_gdf.fillna({'Protest Count':0, 'Riot Count':0, 'VOC Count':0, 'StratDev Count':0, 
#                 'Explosive/Remote Violence Count':0, 'Battles Count':0, 'Event Count':0, 
#                 'ACLED_Fatalities':0}, inplace=True)

fin_gdf.fillna({'Protest Count':0, 'Event Count':0, 
                'ACLED_Fatalities':0}, inplace=True)

#Alternatively could fill nas by means (doesn't make sense to me) - have to do this for spatial z-score calculation to work as nans throw the calculation
# fin_gdf.fillna({'Protest Count':fin_gdf['Protest Count'].mean(), 'Riot Count':fin_gdf['Riot Count'].mean(), 'VOC Count':fin_gdf['Protest Count'].mean(), 'StratDev Count':fin_gdf['Protest Count'].mean(), 
#                 'Explosive/Remote Violence Count':fin_gdf['Protest Count'].mean(), 'Battles Count':fin_gdf['Protest Count'].mean(), 'Event Count':fin_gdf['Protest Count'].mean(), 
#                 'ACLED_Fatalities':fin_gdf['Protest Count'].mean()}, inplace=True)



# COMMAND ----------

fin_gdf.NAME_2

# COMMAND ----------

####
#Weights (Google Contiguity and Spatial Associaton for more info - also pysal's documentation and user example was used heavily for this script)
####
#Queen contiguity
wq = lps.weights.Queen.from_shapefile(filepath='adm2/SDN_adm2.shp')
# wq.transform = 'r'

#KNN
wk= lps.weights.KNN.from_shapefile(filepath='adm2/SDN_adm2.shp', k=5)
# wk.transform='r'

# COMMAND ----------

fin_gdf.tail()

# COMMAND ----------

#Set varlist
# continued
varlist = ['Protest Count', 'Event Count']

#invalid value in battles var - check acled data source - UPDATE solved - due to no battles happenng 2020 it's just a divide by zero - results for 2020 null
#Calculate G* z-scores
#For contigutiy (queen wieghts) set transform parameter to 'R' (source: Arcgis documentation)
for var in varlist:
    print(var)
    #df = fin_gdf.copy()
    #df[var].dropna(inplace=True)
    # G = G_Local(fin_gdf[var], wq, star=True, permutations=999)
    G = G_Local(fin_gdf['Event Count'], wk, transform='r', permutations=999)
    fin_gdf[var+'_Gzs'] = G.Zs
    fin_gdf[var+'_Gpsim'] = G.p_sim

# COMMAND ----------

#viz - in case one wants to check results / or play with confidence levels before deploying
import matplotlib.pyplot as plt
from descartes import PolygonPatch
BLUE = '#6699cc'
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import cm
#Viz results - #G_psim is the pvalue threshold (e.g. .10 = 90% confidence interval)
## Genearte maps in map directory
df = fin_gdf.copy()
conditions = [
    (df['Event Count_Gpsim'] < 0.10) & (df['Event Count_Gzs'] > 0),
    (df['Event Count_Gpsim'] < 0.10) & (df['Event Count_Gzs'] < 0),
    (df['Event Count_Gpsim'] > 0.10)
     ]

#Can ignore this - is just for visualizing with ggplot/python but I am sure there's a better way to do it with tableau
#Essentially created dummy variables based on significance level to then visualize
choices = [1,2,0]

df['viz'] = np.select(conditions, choices)

legend_elements = [   Line2D([0], [0], marker='o', color='w', label='Cold',
                          markerfacecolor='Blue', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Hot',
                          markerfacecolor='Red', markersize=10),
                    Line2D([0], [0], marker='o', color='w', label='Not significant',
                          markerfacecolor='Grey', markersize=10)]


#Static map here
from matplotlib import colors
hmap = colors.ListedColormap([ 'lightgrey', 'red', 'blue'])
f, ax = plt.subplots(1, figsize=(9, 9))
df.assign(cl=df['viz']).plot(column='cl', categorical=True, \
        k=2, cmap=hmap, linewidth=0.1, ax=ax, \
        edgecolor='white')
ax.legend(handles=legend_elements, loc='upper right')        
ax.set_axis_off()
#plt.title("Protest Hot and Cold Zones (ACLED 2020-2021)")
plt.show()

# COMMAND ----------



# COMMAND ----------


