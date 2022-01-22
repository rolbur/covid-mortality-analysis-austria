import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
from matplotlib import pyplot

###############################################################################
# input
###############################################################################

mortality_rate_file = "OGD_rate_kalwo_GEST_KALWOCHE_STR_100.csv" # official mortality data: https://www.data.gv.at/katalog/dataset/6f529d58-9a21-3da9-8c33-d191178d7657
covid_cases_file = "CovidFaelle_Altersgruppe.csv" # official data about covid cases and deaths https://www.data.gv.at/katalog/dataset/3765ed62-0f9d-49ad-83b0-1405ed833108
vacc_doses_file = "COVID19_vaccination_doses_timeline.csv" # official data about vaccination progress https://www.data.gv.at/katalog/dataset/276ffd1e-efdd-42e2-b6c9-04fb5fa2b7ea

mortalityBaselineYear = 2019 # for analysis of covid deaths vs. excess mortality of covid years
mortalityBaselineYear_2 = 2020 # for analysis of excess mortality (with already deduced covid deaths) vs. vaccination progress

###############################################################################
# main
###############################################################################

#import data
data_vaccDoses_df = pd.read_csv(vacc_doses_file, sep=";")
data_mortalRate_df = pd.read_csv(mortality_rate_file, sep=";", decimal=",")
data_covidCases_df = pd.read_csv(covid_cases_file, sep=";", decimal=",")

#data preparation
#convert times in datasets to valid datetime format, add to column "Time"
data_vaccDoses_df["Time"] = [datetime.datetime.strptime(date[0:10], "%Y-%m-%d") for date in data_vaccDoses_df["date"].to_numpy(dtype=str)]
data_covidCases_df["Time"] = pd.to_datetime(data_covidCases_df["Time"],dayfirst=True)
# due to KW issue, assign last week of 2020 to 2021
years = []
kws = []
times = []
for date in data_mortalRate_df["C-KALWOCHE-0"].to_numpy(dtype=str):
    _time = datetime.datetime.strptime(date+"1", "KALW-%Y%W%w")
    _year = int(date[5:9])
    _kw = int(date[9:11])
    if (_year==2020 and _kw==53):
        years.append(2021)
        kws.append(1)
        times.append(_time+datetime.timedelta(weeks=1))
    elif (_year==2021):
        years.append(_year)
        kws.append(_kw+1)
        times.append(_time+datetime.timedelta(weeks=2))
    else:
        years.append(_year)
        kws.append(_kw)
        times.append(_time+datetime.timedelta(weeks=1))
data_mortalRate_df["Year"] = years
data_mortalRate_df["KW"] = kws
data_mortalRate_df["Time"] = times
#for mortality data, filter out unused data (e.g. sex, states)
data_mortalRate_df_filtered = data_mortalRate_df.loc[(data_mortalRate_df["C-BLWO-0"] == "BLWO-0") & \
                                                     (data_mortalRate_df["C-SEXWO-0"] == "SEXWO-0")].copy()
#out of data_covidCases_df create new dataframe with cumulated covid deaths
data_covidCases_df_filtered = data_covidCases_df.loc[data_covidCases_df["Bundesland"]=="Österreich"]
covidDeathsCum_dict = defaultdict(list)
for groupName, groupData in data_covidCases_df_filtered.groupby("Time"):
    covidDeathsCum_dict["Time"].append(groupName)
    covidDeathsCum_dict["CovidDeathsCum"].append(groupData["AnzahlTot"].sum())
covidDeathsCum_df = pd.DataFrame(covidDeathsCum_dict)
#create new dataframes about mortalility filtered for some defined years: baseline year and covid years
data_mortalRate_df_filtered_baseline = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([mortalityBaselineYear]))].copy()
data_mortalRate_df_filtered_CovidYears = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([2020,2021]))].copy()
# add information about cumulated excess mortality to dataframe
mortalBaseline_np = data_mortalRate_df_filtered_baseline["F-ANZ-1"].to_numpy()
mortal_np = data_mortalRate_df_filtered_CovidYears["F-ANZ-1"].to_numpy()
excessmortal = mortal_np - np.tile(mortalBaseline_np,[2])
data_mortalRate_df_filtered_CovidYears["excessmortalCum"] = excessmortal.cumsum()

# clean mortalility data, exclude covid deaths, add coved death deduced data to dataframe
F_ANZ_cleanedByCovidDeaths = []
F_ANZ_cleanedByCovidDeaths.append(data_mortalRate_df_filtered["F-ANZ-1"].to_numpy(dtype=int)[0]) #add last value
for lb, ub, F_ANZ in zip(data_mortalRate_df_filtered["Time"].to_numpy()[0:-1], \
                  data_mortalRate_df_filtered["Time"].to_numpy()[1::] , \
                  data_mortalRate_df_filtered["F-ANZ-1"].to_numpy(dtype=int)[1::]):
    value_lb = np.interp(float(lb), covidDeathsCum_df["Time"].to_numpy(dtype=float), covidDeathsCum_df["CovidDeathsCum"].to_numpy(), left=0)
    value_ub = np.interp(float(ub), covidDeathsCum_df["Time"].to_numpy(dtype=float), covidDeathsCum_df["CovidDeathsCum"].to_numpy(), left=0)
    delta = value_ub-value_lb
    F_ANZ_cleanedByCovidDeaths.append(F_ANZ-delta)
#F_ANZ_cleanedByCovidDeaths.append(data_mortalRate_df_filtered["F-ANZ-1"].to_numpy(dtype=int)[-1]-delta) #add last value
data_mortalRate_df_filtered["F_ANZ_cleanedByCovidDeaths"] = F_ANZ_cleanedByCovidDeaths
# create some interesting views from covid mortality dataframe for later comparison
data_mortalRate_df_filtered_CovidYearsCleaned = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([2020,2021]))].copy()
data_mortalRate_df_filtered_LastYears = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([2016,2017,2018,2019,2020,2021]))].copy()

#calculate excess mortalility based on cleaned dataset (deduced covid deaths) for analysis of vaccination related deaths
data_mortalRate_df_filtered_baseline2 = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([mortalityBaselineYear_2]))]
data_mortalRate_df_filtered_vaccYear = data_mortalRate_df_filtered.loc[(data_mortalRate_df_filtered["Year"].isin([2021]))].copy()
excessMort2_df = pd.DataFrame()
mortalBaseline_np = data_mortalRate_df_filtered_baseline2["F_ANZ_cleanedByCovidDeaths"].to_numpy()
mortal_np = data_mortalRate_df_filtered_vaccYear["F_ANZ_cleanedByCovidDeaths"].to_numpy()
excessmortal = mortal_np - mortalBaseline_np
data_mortalRate_df_filtered_vaccYear["excessmortalCum"] = excessmortal.cumsum()
        
#%%############################################################################
# plot raw data
###############################################################################
# plot mortalility per year
fig, ax = pyplot.subplots()
for groupName, groupData in data_mortalRate_df_filtered_LastYears.groupby(["Year"]):
    groupData["cum"] = groupData["F-ANZ-1"].to_numpy().cumsum()
    groupData.plot(x="KW", y="cum", label=groupName, ax=ax, ylim=[0, 90000])
ax.grid(True)
ax.set_title("mortality of last years")

# plot covid deaths
fig, ax = pyplot.subplots()
covidDeathsCum_df.plot(x="Time", y="CovidDeathsCum", ylim=[0, 18000], ax=ax)
ax.grid(True)
ax.set_title("cumulated covid deaths")

# plot vaccine progression
fig, ax = pyplot.subplots()
for i,[groupName, groupData] in enumerate(data_vaccDoses_df.loc[data_vaccDoses_df["state_name"]=="Österreich"].groupby(["vaccine","dose_number"])):
    groupData.plot(x="Time", y="doses_administered_cumulative", label = f"{groupName[0]} {groupName[1]}", ax=ax, ylim=[0,8e6])
ax.grid(True)
ax.set_title("vaccination progress")

#%%############################################################################
# plot excess mortalility vs. covid deaths
###############################################################################
# init plot
fig, ax = pyplot.subplots()
# plot covid deaths
covidDeathsCum_df.plot(x="Time", y="CovidDeathsCum", ylim=[0,19000], ax=ax)
# plot excess mortalility
data_mortalRate_df_filtered_CovidYears.plot(x="Time", y="excessmortalCum", color = ['#BB0000'], 
                                            label = f"excess mortality, cumulated, baseline {mortalityBaselineYear}", 
                                            ax=ax)
ax.grid(True)
ax.set_title("excess mortality vs. covid deaths")

#%%############################################################################
# plot cleaned morabilities
###############################################################################
# plot mortalility per year
fig, ax = pyplot.subplots()
for groupName, groupData in data_mortalRate_df_filtered_LastYears.groupby(["Year"]):
    groupData["cum"] = groupData["F_ANZ_cleanedByCovidDeaths"].to_numpy().cumsum()
    groupData.plot(x="KW", y="cum", label=groupName, ax=ax, ylim=[0, 90000])
ax.grid(True)
ax.set_title("mortality, deduced covid deaths")

#%%############################################################################
# plot excess mortalility of vacc year, compare cleaned data (excluded covid deaths)
###############################################################################
# init plot
fig, ax = pyplot.subplots()
# plot excess mortalility
data_mortalRate_df_filtered_vaccYear.plot(x="Time", y="excessmortalCum", 
                                          label = f"excess mortality, Covid deaths deducted, cumulated, baseline {mortalityBaselineYear_2}", 
                                          color = ['b'], marker="o", ax=ax)
ax.legend(loc="upper left")
ax2 = ax.twinx()
# plot vaccine progression
for i,[groupName, groupData] in enumerate(data_vaccDoses_df.loc[data_vaccDoses_df["state_name"]=="Österreich"].groupby(["vaccine","dose_number"])):
    groupData.plot(x="Time", y="doses_administered_cumulative", label = f"{groupName[0]} {groupName[1]}", ax=ax2, ylim=[0,8e6])
ax.grid(True)
ax.set_title("excess mortality (deduced covid deaths) vs. vaccination progress")
