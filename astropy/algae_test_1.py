
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
import plotly_express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import scale

def ret_year(date_yr):
    years = []
    for d in date_yr:
        date_y = dt.strptime(d, "%b-%y")
        date_n = dt.strftime(date_y, "%y")
        # print(date_n)
        years.append(date_n)
    date_se = pd.Series(years)
    return(date_se)

def make_df(month_list):
    # Takes in the list and return a pandas dataframe, with Dates and Years as
    # separate columns
    
    df_temp = pd.DataFrame(month_list, index=None, columns=['Date', 'Al_C'])
    # Good Groupby Tut -> https://realpython.com/pandas-groupby/
    df_temp = df_temp.groupby("Date", sort=False, as_index=False).mean()
    df_temp = pd.eval("Year = ret_year(df_temp.Date)", target=df_temp)

    return df_temp


cols = ['Time', 'Algae Conc']
df = pd.read_csv("WIBI84_21.csv", index_col='Time')

df_copy = df.copy()
df_copy.rename(columns={'constant': 'Algae Concentration'}, inplace=True)


time = df_copy.index

df_c = df_copy.copy()
alg = df_c['Algae Concentration']
algae = []
for a in alg:
    if str(a) == 'nan':
        x = np.float64(0)
    else:
        x = np.float64(a)
    algae.append(x)

df_copy['Algae Con'] = algae

df_copy = df_copy.drop('Algae Concentration', axis = 1)


time_series = []
march = []
april = []
may = []
june = []
july = []
august = []
september = []
october = []


# For info regarding how to use strptime
# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
for t in time:
    # print(t)
    dat = dt.strptime(t, "%d-%b-%y")
    time_s = dt.strftime(dat, "%b-%y")
    if dat.month == 3:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        march.append(time_val)
    
    elif dat.month == 4:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        april.append(time_val)
    
    elif dat.month == 5:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        may.append(time_val)

    elif dat.month == 6:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        june.append(time_val)
    
    elif dat.month == 7:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        july.append(time_val)
    
    elif dat.month == 8:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        august.append(time_val)

    elif dat.month == 9:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        september.append(time_val)
    
    elif dat.month == 10:
        val = df_copy.loc[t]['Algae Con']
        time_val = [time_s, val]
        october.append(time_val)

    # time_s = '{}/{}'.format(dat.month, dat.year)
    time_series.append(time_s)

time = pd.Series(time_series)
df_copy.index = time

df_copy.index.rename('Date', inplace=True)


df_march = make_df(march)
df_april = make_df(april)
df_may = make_df(may)
df_june = make_df(june)
df_july = make_df(july)
df_august = make_df(august)
df_september = make_df(september)


ax = plt.gca()

# df_copy.plot(y='Algae Con', color='red', ax=ax, figsize=(15, 10), grid=True)
df_march.plot(x='Year', y='Al_C', ax=ax, color='blue', label='March', figsize=(30, 20))
df_april.plot(x='Year', y='Al_C', ax=ax, color='red', label='April')
df_may.plot(x='Year', y='Al_C', ax=ax, color='purple', label='May')
df_june.plot(x='Year', y='Al_C', ax=ax, color='yellow', label='June')
df_july.plot(x='Year', y='Al_C', ax=ax, color='orange', label='July')
df_august.plot(x='Year', y='Al_C', ax=ax, color='cyan', label='August')
df_september.plot(x='Year', y='Al_C', ax=ax, color='maroon', label='September')


df_merged = pd.merge(df_may, df_march, how='left', on='Year', suffixes=('_May', '_Mar'))
df_merged = pd.merge(df_merged, df_april, how='left', on='Year', suffixes=('_Mar', '_Apr'))
df_merged = pd.merge(df_merged, df_june, how='left', on='Year', suffixes=('_Apr', '_Jun'))
df_merged = pd.merge(df_merged, df_july, how='left', on='Year', suffixes=('_Jun', '_Jul'))
df_merged = pd.merge(df_merged, df_august, how='left', on='Year', suffixes=('_Jul', '_Aug'))
df_merged = pd.merge(df_merged, df_september, how='left', on='Year', suffixes=('_Aug', '_Sep'))

df_merged.rename(columns={'Al_C': 'Al_C_Sep'}, inplace=True)
df_merged.plot(x="Year", y=['Al_C_Mar', 'Al_C_Apr', 'Al_C_May', 'Al_C_Jun', 'Al_C_Jul', 'Al_C_Aug', 'Al_C_Sep'])


# df_october.plot(x='Year', y='Algae Conc', ax=ax, color='teal', label='October')
plt.title("NCBC ALGAE CONCENTRATION (1988-2021)", fontsize=40)
plt.xlabel("Year", fontsize=20)
plt.ylabel("Algae Concentration", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.savefig("Algae_Concentration(88-22).png")
plt.close()

df_merged = df_merged.fillna(np.float64(0))
fig = px.line(df_merged, x='Year', y=['Al_C_Mar', 'Al_C_Apr', 'Al_C_May', 'Al_C_Jun', 'Al_C_Jul', 'Al_C_Aug', 'Al_C_Sep'])

buttonlist = ['Al_C_Mar', 'Al_C_Apr', 'Al_C_May', 'Al_C_Jun', 'Al_C_Jul', 'Al_C_Aug', 'Al_C_Sep']

fig = go.Figure()
buttonlist = ['Al_C_Mar', 'Al_C_Apr', 'Al_C_May', 'Al_C_Jun', 'Al_C_Jul', 'Al_C_Aug', 'Al_C_Sep']
fig = px.line(df_merged, x='Year', y=['Al_C_Mar', 'Al_C_Apr', 'Al_C_May', 'Al_C_Jun', 'Al_C_Jul', 'Al_C_Aug', 'Al_C_Sep'], range_y=(0,270))
fig.update_layout(
    title="Test data",
    yaxis_title="Algae Concentration",
    xaxis_title="Year",
    height = 400,
    width=800,
    # Add dropdown
    updatemenus=[
        dict(
            active=0,
            buttons=list([
                dict(label="All Plots",
                    method="update",
                    args=[{"visible":[True, True, True, True, True, True, True]},
                            {"title":"All Months"}]),
                dict(label="MARCH",
                    method="update",
                    args=[{"visible":[True, False, False, False, False, False, False]},
                            {"title":"March"}]),
                dict(label="April",
                    method="update",
                    args=[{"visible":[False, True, False, False, False, False, False]},
                            {"title":"April"}]),
                dict(label="May",
                    method="update",
                    args=[{"visible":[False, False, True, False, False, False, False]},
                            {"title":"May"}]),
                dict(label="June",
                    method="update",
                    args=[{"visible":[False, False, False, True, False, False, False]},
                            {"title":"June"}]),
                dict(label="July",
                    method="update",
                    args=[{"visible":[False, False, False, False, True, False, False]},
                            {"title":"July"}]),
                dict(label="August",
                    method="update",
                    args=[{"visible":[False, False, False, False, False, True, False]},
                            {"title":"August"}]),
                dict(label="September",
                    method="update",
                    args=[{"visible":[False, False, False, False, False, False, True]},
                            {"title":"September"}]),
            ])
        )
    ],
    autosize=True
)

fig.show()
