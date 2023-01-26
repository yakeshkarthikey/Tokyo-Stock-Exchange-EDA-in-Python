import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from tensorflow import keras
from sklearn.metrics import *
import datetime
import plotly
import plotly.graph_objects as go
import plotly.express as px
pd.set_option('display.max_columns',10)


#open dataset and anlysing
data = pd.read_csv("C:/Users/YK/PycharmProjects/Py/JPX Tokyo Stock Exchange Prediction/stock_list.csv")
df = pd.DataFrame(data)
print(df)
print(data.describe())
print(data.isnull().sum())


#creating a datafram with no null values
newdf = df.dropna()
print(newdf.isnull().sum())
print(newdf.describe())

#print each column values individually

sec_code = newdf['SecuritiesCode']
eff_date = newdf['EffectiveDate']
name =newdf['Name']
s_p = newdf['Section/Products']
market_seg = newdf['NewMarketSegment']
Sector_33_Code = newdf['33SectorCode']
Sector_33_Name = newdf['33SectorName']
Sector_17_Code = newdf['17SectorCode']
Sector_17_Name = newdf['17SectorName']
NewIndexSeriesSizeCode = newdf['NewIndexSeriesSizeCode']
NewIndexSeriesSize = newdf['NewIndexSeriesSize']
TradeDate = newdf['TradeDate']
Close = newdf['Close']
IssuedShares = newdf['IssuedShares']
MarketCapitalization = newdf['MarketCapitalization']
Universe0 = newdf['Universe0']


print("securitiescode")
print(sec_code.head())
print("#"*100)

print("eff_date")
print(eff_date.head())
print("#"*100)


print("stock-name")
print(name.head())
print("#"*100)

print("service/product")
print(s_p.head())
print("#"*100)

print("newmarketsegment")
print(market_seg.head())
print("#"*100)

print("Sector_33_Code")
print(Sector_33_Code.head())
print("#"*100)

print("Sector_33_Name")
print(Sector_33_Name.head())
print("#"*100)

print("Sector_17_Code")
print(Sector_17_Code.head())
print("#"*100)

print("Sector_17_Name")
print(Sector_17_Name.head())
print("#"*100)

print("newIndex_series_size_code")
print(NewIndexSeriesSizeCode.head())
print("#"*100)

print("new_index_size")
print(NewIndexSeriesSize.head())
print("#"*100)

print("Tradedate")
print(TradeDate.head())
print("#"*100)

print("Close")
print(Close.head())
print("#"*100)

print("issued_shares")
print(IssuedShares.head())
print("#"*100)

print("Marketcapitalization")
print(MarketCapitalization.head())
print("#"*100)



#convert date into its equivalent format

newdf['EffectiveDate'] = pd.to_datetime(newdf['EffectiveDate'], format='%Y%m%d')
newdf['TradeDate'] = pd.to_datetime(newdf['TradeDate'],format='%Y%m%d')

date = newdf['EffectiveDate']
t_date = newdf['TradeDate']
print(newdf)

#dataframe for Sector_33
df1 = pd.concat([sec_code,name,date,Sector_33_Name,Sector_33_Code,MarketCapitalization,IssuedShares],axis=1,ignore_index=True)
df1.columns = ['securitycode','stock-name','date','Sector_33_Name','Sector_33_Code','market-capitalization','shares']
print(df1)

#dataframe for Sector_17
df2 = pd.concat([sec_code,name,date,Sector_17_Name,Sector_17_Code,MarketCapitalization,IssuedShares],axis=1,ignore_index=True)
df2.columns = ['securitycode','stock-name','date','Sector_17_Name','Sector_17_Code','market-capitalization','shares']
print(df2)

#data_visualization

plt.bar(Sector_33_Name[:500:2],MarketCapitalization[:500:2])
plt.title("sector_33_name vs Market_capitalization")
plt.xlabel("sector_33_name")
plt.ylabel("Market_capitalization")
plt.show()

plt.scatter(Sector_17_Name[:500:2],MarketCapitalization[:500:2])
plt.title("sector_17_name vs Market_capitalization")
plt.xlabel("sector_17_name")
plt.ylabel("Market_capitalization")

plt.show()

#bar_graph
fig = px.bar(df1, x="Sector_33_Name", y="shares",color = "market-capitalization")
fig.show()

fig = px.bar(df2, x="Sector_17_Name", y="shares",color = "market-capitalization")
fig.show()


# plotting pie chart
fig = go.Figure(data=[go.Pie(labels=Sector_33_Name,
                             values=IssuedShares,title='Sector_17_Name Vs IssuedShares')])
fig.show()

# plotting pie chart
fig = go.Figure(data=[go.Pie(labels=Sector_17_Name,
                             values=IssuedShares,title='Sector_17_Name Vs IssuedShares')])
fig.show()


#train_dataset
stock_details = pd.read_csv('C:/Users/YK/PycharmProjects/Py/JPX Tokyo Stock Exchange Prediction/stock_prices.csv')
print(stock_details.columns)
stock_details = pd.DataFrame(stock_details)


df3 = pd.concat([stock_details['SecuritiesCode'],stock_details['Open'],stock_details['Close'],
                 stock_details['Low'],stock_details['High'],stock_details['Target']],axis=1)

#removing_NULL_values
print(df3.describe())
print(df3.isnull().sum())

df3 = df3.dropna()
print(df3.isnull().sum())
print(df3.describe())
print(df3)


print(df3.info())


#applying_MinMaxScaler to scale the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

open = MinMaxScaler().fit_transform(df3['Open'].values.reshape(-1,1))
high = MinMaxScaler().fit_transform(df3['High'].values.reshape(-1,1))
low = MinMaxScaler().fit_transform(df3['Low'].values.reshape(-1,1))
close = MinMaxScaler().fit_transform(df3['Close'].values.reshape(-1,1))
target = MinMaxScaler().fit_transform(df3['Target'].values.reshape(-1,1))


print(open)
print("_"*100)
print(close)

x_train,x_test,y_train,y_test = train_test_split(open,close,train_size=0.75,random_state=0)

print("X_train:",x_train)
print("y_train",y_train)
print("X_test:",x_test)
print("y_test",y_test)

