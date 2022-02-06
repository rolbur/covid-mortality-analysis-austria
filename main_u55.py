import pandas as pd
import numpy as np
import datetime
from collections import defaultdict
from matplotlib import pyplot

###############################################################################
# input
###############################################################################

mortality_rate_file = "OGD_gest_kalwo_alter_GEST_KALWOCHE_5J_100.csv" # official mortality data: https://www.data.gv.at/katalog/dataset/stat_gestorbene-in-osterreich-ohne-auslandssterbefalle-ab-2000-nach-kalenderwoche-und-5-jahresg
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
#for mortality data, filter out unused data (e.g. sex, states), only consider those until 55 years
data_mortalRate_df_filtered = data_mortalRate_df.loc[(data_mortalRate_df["C-ALTER5-0"].isin(['ALTER5-1','ALTER5-2','ALTER5-3','ALTER5-4','ALTER5-5','ALTER5-6','ALTER5-7','ALTER5-8','ALTER5-9','ALTER5-10','ALTER5-11']))].copy()
mortalRate_filtered_dict = defaultdict(list)
for groupName, groupData in data_mortalRate_df_filtered.groupby(["Year","KW"]):
    mortalRate_filtered_dict["F-ANZ-1"].append(groupData["F-ANZ-1"].sum())
    mortalRate_filtered_dict["Time"].append(groupData["Time"].unique()[0])
    mortalRate_filtered_dict["KW"].append(groupData["KW"].unique()[0])
    mortalRate_filtered_dict["Year"].append(groupData["Year"].unique()[0])
data_mortalRate_df_filtered = pd.DataFrame(mortalRate_filtered_dict)
                                                      
#out of data_covidCases_df create new dataframe with cumulated covid deaths
data_covidCases_df_filtered = data_covidCases_df.loc[(data_covidCases_df["Bundesland"]=="Österreich") & \
                                                     (data_covidCases_df["AltersgruppeID"]<=6)]
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

# clean mortalility data, exclude covid deaths, add covid death deduced data to dataframe
F_ANZ_cleanedByCovidDeaths = []
F_ANZ_cleanedByCovidDeaths.append(data_mortalRate_df_filtered["F-ANZ-1"].to_numpy(dtype=int)[0]) #add first value manually
for lb, ub, F_ANZ in zip(data_mortalRate_df_filtered["Time"].to_numpy()[0:-1], \
                  data_mortalRate_df_filtered["Time"].to_numpy()[1::] , \
                  data_mortalRate_df_filtered["F-ANZ-1"].to_numpy(dtype=int)[1::]):
    value_lb = np.interp(float(lb), covidDeathsCum_df["Time"].to_numpy(dtype=float), covidDeathsCum_df["CovidDeathsCum"].to_numpy(), left=0)
    value_ub = np.interp(float(ub), covidDeathsCum_df["Time"].to_numpy(dtype=float), covidDeathsCum_df["CovidDeathsCum"].to_numpy(), left=0)
    delta = value_ub-value_lb
    F_ANZ_cleanedByCovidDeaths.append(F_ANZ-delta)
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
    groupData.plot(x="KW", y="cum", label=groupName, ax=ax, ylim=[0, 8000])
ax.grid(True)
ax.set_title("mortality of last years")
ax.set_ylabel("deaths")

# plot covid deaths
fig, ax = pyplot.subplots()
covidDeathsCum_df.plot(x="Time", y="CovidDeathsCum", ylim=[0, 400], ax=ax)
ax.grid(True)
ax.set_title("cumulated covid deaths")
ax.set_ylabel("deaths")

# plot vaccine progression
fig, ax = pyplot.subplots()
for groupName, groupData in data_vaccDoses_df.loc[data_vaccDoses_df["state_name"]=="Österreich"].groupby(["vaccine","dose_number"]):
    groupData.plot(x="Time", y="doses_administered_cumulative", label = f"{groupName[0]} {groupName[1]}", ax=ax, ylim=[0,8e6])
ax.grid(True)
ax.set_title("vaccination progress")
ax.set_ylabel("vaccine doses")

#%%############################################################################
# plot excess mortalility vs. covid deaths
###############################################################################
# init plot
fig, ax = pyplot.subplots()
# plot covid deaths
covidDeathsCum_df.plot(x="Time", y="CovidDeathsCum", ylim=[-150,400], ax=ax)
# plot excess mortalility
data_mortalRate_df_filtered_CovidYears.plot(x="Time", y="excessmortalCum", color = ['#BB0000'], 
                                            label = f"excess mortality, cumulated, baseline {mortalityBaselineYear}", 
                                            ax=ax)
ax.grid(True)
ax.set_title("excess mortality vs. covid deaths")
ax.set_xlim(left = datetime.datetime(2020,1,1), right = datetime.datetime(2021,12,31))
ax.set_ylabel("deaths")

#%%############################################################################
# plot cleaned mortabilities
###############################################################################
# plot mortalility per year
fig, ax = pyplot.subplots()
#for groupName, groupData in data_mortalRate_df_filtered_LastYears.loc[(data_mortalRate_df_filtered_LastYears["Year"].isin([2019,2020,2021]))].groupby(["Year"]):
for groupName, groupData in data_mortalRate_df_filtered_LastYears.groupby(["Year"]):
    groupData["cum"] = groupData["F_ANZ_cleanedByCovidDeaths"].to_numpy().cumsum()
    groupData.plot(x="KW", y="cum", label=groupName, ax=ax, ylim=[0, 8000])
ax.grid(True)
ax.set_title("mortality, deduced covid deaths")
ax.set_ylabel("deaths")

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
ax.grid(True)
ax2 = ax.twinx()
# plot vaccine progression
for groupName, groupData in data_vaccDoses_df.loc[data_vaccDoses_df["state_name"]=="Österreich"].groupby(["vaccine","dose_number"]):
    groupData.plot(x="Time", y="doses_administered_cumulative", label = f"{groupName[0]} {groupName[1]}", ax=ax2, ylim=[0,8e6])

ax.set_title("excess mortality (deduced covid deaths) vs. vaccination progress")
ax.set_xlim(left = datetime.datetime(2021,1,1), right = datetime.datetime(2021,12,31))
ax.set_ylabel("deaths")
ax2.set_ylabel("vaccine doses")
