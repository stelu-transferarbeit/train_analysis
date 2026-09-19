from pathlib import Path
from typing import Annotated

import geopandas
import matplotlib.cm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from cyclopts import App, Parameter
from numpy import log, ones_like, triu
from rich import print
from rich.markdown import Markdown
from statsmodels.iolib.summary import SimpleTable

from train_analysis.analysis import (
    panel,
    pooled,
    random_effects,
)

app = App()


eurostat_base = (
    "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT"
)
time_filter = "c[TIME_PERIOD]=2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008"
country_filter = (
    "c[geo]=BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,IT,LV,LT,HU,NL,AT,PL,PT,RO,SI,SK,FI,SE,NO,CH"
)
format_options = "compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL"
common_filters = f"{time_filter}&{country_filter}&{format_options}"
datasets = {
    "rail_length": f"{eurostat_base}/rail_if_line_na/1.0/*.*.*.*.*?c[freq]=A&c[unit]=KM&c[tra_infr]=TOTAL,RL_ELC&c[tra_meas]=FR_ONL,TOTAL&{common_filters}",
    "cars_capita": f"{eurostat_base}/road_eqs_carhab/1.0/*.*.*?c[freq]=A&c[unit]=NR&{common_filters}",
    "cars": f"{eurostat_base}/road_eqs_carmot/1.0/*.*.*.*.*?c[freq]=A&c[unit]=NR&c[mot_nrg]=TOTAL&c[engine]=TOTAL&{common_filters}",
    "rail_passengers": f"{eurostat_base}/rail_pa_total/1.0/*.*.*?c[freq]=A&c[unit]=MIO_PKM&{common_filters}",
    "population": f"{eurostat_base}/demo_pjan/1.0/*.*.*.*.*?c[freq]=A&c[unit]=NR&c[age]=TOTAL&c[sex]=T&{common_filters}",
    "gdp_per_capita": f"{eurostat_base}/sdg_08_10/1.0/*.*.*.*?c[freq]=A&c[unit]=CLV20_EUR_HAB&c[na_item]=B1GQ&{common_filters}",
    "gdp": f"{eurostat_base}/nama_10_gdp/1.0/*.*.*.*?c[freq]=A&c[unit]=CP_MEUR&c[na_item]=B1GQ&{common_filters}",
    "area": f"{eurostat_base}/reg_area3/1.0/*.*.*.*?c[freq]=A&c[landuse]=TOTAL&c[unit]=KM2&{country_filter}&c[TIME_PERIOD]=2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024&{format_options}",
    # "modal_split": f"{eurostat_base}/tran_hv_ms_psmod/1.0?c[vehicle]=TRN,CAR,BUS_TOT,AC&{common_filters}",
    # "rail_investment": "https://sdmx.oecd.org/public/rest/data/OECD.ITF,DSD_INFRINV@DF_INFRINV,1.0/.A..EUR.TOT_INL+MAR+AIR.Q",
    "rail_accidents": f"{eurostat_base}/tran_sf_railac/1.0/*.*.*.*?c[freq]=A&c[unit]=NR&c[accident]=TOTAL&{common_filters}",
    # "rail_high_speed": f"{eurostat_base}/rail_if_line_sp/1.0/*.*.*.*?c[freq]=A&c[tra_infr]=TOTAL,RL_DHSPD,RL_UHSPD&c[unit]=KM&{common_filters}",
}


def transform_area(data: pd.DataFrame):
    countries = []
    for country, group in data.groupby("geo"):
        filled = pd.DataFrame({"TIME_PERIOD": list(range(2008, 2013))})
        filled["geo"] = country
        filled["OBS_VALUE"] = None
        filled["OBS_VALUE"] = filled["OBS_VALUE"].astype("float64")
        countries.append(
            pd.concat([filled, group.drop("landuse", axis="columns")]).bfill()
        )

    return pd.concat(countries)


def transform_cars(data: pd.DataFrame):
    return data.drop(["mot_nrg", "engine"], axis="columns")


def transform_rail_length(data: pd.DataFrame):
    pass_data = data[data["tra_meas"] == "Total"].merge(
        data[data["tra_meas"] == "Freight only"],
        on=["tra_infr", "geo", "TIME_PERIOD"],
        suffixes=("", "_freight"),
    )
    pass_data["OBS_VALUE"] -= pass_data["OBS_VALUE_freight"].fillna(0)
    pass_data = pass_data.loc[
        :,
        ~pass_data.columns.isin(["OBS_VALUE_freight", "tra_meas_freight", "tra_meas"]),
    ]
    pass_data = pass_data[pass_data["tra_infr"] == "Total"].merge(
        pass_data[pass_data["tra_infr"] == "Electrified railway lines"],
        on=["geo", "TIME_PERIOD"],
        suffixes=("", "_electrified"),
    )
    pass_data["OBS_VALUE_electrified"] = (
        pass_data["OBS_VALUE_electrified"] / pass_data["OBS_VALUE"]
    )
    return pass_data.rename(
        {
            "OBS_VALUE": "total_rail_length",
            "OBS_VALUE_electrified": "rail_electrification_share",
        },
        axis="columns",
    ).loc[
        :,
        ~pass_data.columns.isin(["tra_infr_electrified", "tra_infr"]),
    ]


def transform_modal_split(data: pd.DataFrame):
    # Interpolate data for France in 2010, 2011, 2012 since the data is clearly wrong.
    data.loc[
        (data["geo"] == "France") & data["TIME_PERIOD"].isin([2010, 2011, 2012]),
        "OBS_VALUE",
    ] = None
    for v in data["vehicle"].unique():
        mask = (data["geo"] == "France") & (data["vehicle"] == v)
        data.loc[mask, "OBS_VALUE"] = data.loc[mask, "OBS_VALUE"].interpolate()
    vehicle_data = None
    for (v,), g in data.groupby(["vehicle"]):
        group_data = g[["geo", "TIME_PERIOD", "OBS_VALUE"]].rename(
            {"OBS_VALUE": v}, axis="columns"
        )
        if vehicle_data is None:
            vehicle_data = group_data
        else:
            vehicle_data = vehicle_data.merge(group_data, on=["geo", "TIME_PERIOD"])
    return vehicle_data


def transform_population(data: pd.DataFrame):
    data["OBS_VALUE"] /= 10_000
    return data


def transform_rail_high_speed(data: pd.DataFrame):
    hsp_data = data[data["tra_infr"] == "Total"].merge(
        data[data["tra_infr"] == "Dedicated high speed railway lines"],
        on=["geo", "TIME_PERIOD"],
        suffixes=("", "_dedicated"),
    )
    hsp_data["OBS_VALUE_dedicated"] /= hsp_data["OBS_VALUE"]
    hsp_data = hsp_data.merge(
        data[data["tra_infr"] == "Upgraded high speed railway lines"],
        on=["geo", "TIME_PERIOD"],
        suffixes=("", "_upgraded"),
    )
    hsp_data["OBS_VALUE_upgraded"] /= hsp_data["OBS_VALUE"]
    return hsp_data.rename(
        {
            "OBS_VALUE_dedicated": "dedicated_high_speed_share",
            "OBS_VALUE_upgraded": "upgraded_high_speed_share",
        },
        axis="columns",
    ).loc[
        :,
        ~hsp_data.columns.isin(["tra_infr_dedicated", "tra_infr_upgraded", "tra_infr"]),
    ]


@app.command(alias="dl")
def download(force: bool = False):
    print("Downloading the necessary datasets and saving as parquet")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    for name, url in datasets.items():

        pq_file = data_dir / f"{name}.parquet"
        if not force and pq_file.is_file():
            continue
        print(f'Saving dataset "{name}"')
        df = pd.read_csv(url, compression="gzip")
        df.to_parquet(pq_file, compression="brotli")


def load_data() -> pd.DataFrame:
    data: pd.DataFrame | None = None
    data_dir = Path("data")
    for name in datasets:
        pq_t_file = data_dir / "transformed" / f"{name}.parquet"
        # if pq_t_file.is_file():
        #     return pd.read_parquet(pq_t_file)
        pq_file = data_dir / f"{name}.parquet"
        d = pd.read_parquet(pq_file)
        freqs = d["freq"].unique().tolist()
        assert freqs == ["Annual"] or freqs == ["A"]
        assert len(d["unit"].unique()) == 1
        d = d.loc[
            :,
            ~d.columns.isin(
                [
                    "DATAFLOW",
                    "freq",
                    "LAST UPDATE",
                    "unit",
                    "CONF_STATUS",
                    "OBS_FLAG",
                    "na_item",
                    "accident",
                    "age",
                    "sex",
                ]
            ),
        ]
        transform = globals().get(f"transform_{name}", lambda d: d)
        d = transform(d).rename({"OBS_VALUE": name}, axis="columns")
        pq_t_file.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(pq_t_file, compression="brotli")
        if data is None:
            data = d
        else:
            data = data.merge(d, on=["geo", "TIME_PERIOD"])
    if data is None:
        raise Exception("No datasets to load")
    data = (
        data[~data["TIME_PERIOD"].isin([2020, 2021, 2022])]
        .rename({"TIME_PERIOD": "year"}, axis="columns")
        .set_index(["geo", "year"])
    )
    data["population_log"] = log(data["population"])
    # The rail accidents can be zero, so the log cannot be calculated. A
    # workaround is to add 1 to ensure no 0 appears, but this will introduce a
    # small bias. For simplicity's sake we ignore the issue.
    #
    # cfr: https://arxiv.org/abs/2203.11820

    data["rail_passengers_pop"] = data["rail_passengers"] / data["population"]
    data["rail_passengers_pop_log"] = log(data["rail_passengers_pop"])
    data["rail_accidents_log"] = log(data["rail_accidents"] + 1)
    data["rail_accidents_div_pkm"] = data["rail_accidents"] / data["rail_passengers"]
    data["rail_accidents_div_pkm_log"] = log(1 + data["rail_accidents_div_pkm"])
    data["cars_capita"] = data["cars"] / data["population"]
    data["cars_capita_log"] = log(data["cars_capita"])
    data["cars_log"] = log(data["cars"])
    data["gdp_log"] = log(data["gdp"])
    data["gdp_capita_log"] = log(data["gdp_per_capita"])
    data["railway_density"] = data["total_rail_length"] / data["area"]
    data["railway_density_log"] = log(data["total_rail_length"] / data["area"])
    data.to_parquet(data_dir / "transformed" / "final.parquet")
    return data.dropna()


def load_cached_data():
    data_dir = Path("data")
    return pd.read_parquet(data_dir / "transformed" / "final.parquet")


@app.command
@app.default
def analyze(
    lag: int = 0,
    remove_outliers: Annotated[
        bool, Parameter(name=["--remove_outliers", "-o"])
    ] = False,
):
    download()
    data = add_lag(load_data(), lag)
    if remove_outliers:
        data = remove_accidents_outliers(data)
    # for country_name, g in data.groupby("geo"):
    #     print(f"Data for country: {country_name}")
    #     print(g.describe([]))
    predictors = [
        "rail_accidents_div_pkm_log",
        "rail_electrification_share",
        "cars_capita_log",
        "gdp_capita_log",
        "railway_density_log",
    ]
    variables = ["rail_passengers", *predictors]
    print(data[variables].corr())
    print(data[predictors].corr())
    _new_section("Variable descriptions")
    print(
        data[["rail_passengers_pop_log", *predictors]].describe(
            [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        )
    )
    if lag == 0:
        _new_section("Predictions without lag")
    else:
        _new_section(f"Predictions with {lag} years of lag")
    _new_section("Simple PanelOLS model with all predictors")
    res = pooled(data, "rail_passengers", predictors)
    print(res)
    _new_section("Simple PanelOLS model with rail passenger-km per population (ln)")
    res = pooled(data, "rail_passengers_pop_log", predictors)
    print(res)
    _new_section("Entity Fixed Effects")
    res_entity = panel(data, "rail_passengers_pop_log", predictors, entity_effects=True)
    print(res_entity)
    _new_section("Entity and Time-Fixed Effects")
    res_entity_time = panel(
        data,
        "rail_passengers_pop_log",
        predictors,
        entity_effects=True,
        time_effects=True,
    )
    print(res_entity_time)
    _new_section("Random Effects")
    res_random = random_effects(data, "rail_passengers_pop_log", predictors)
    print(res_random)

    # # Model 1: PooledOLS with rail_accidents as the sole predictor
    # res1 = single_regressor_no_effects(data)
    # print(res1)
    # print(res1.params)
    # # Model 2: PooledOLS with rail_accidents as the predictor and fixed entity effects
    # res2 = single_regressor_entity_fixed_effects(data)
    # print(res2)
    # print(res2.params)
    # # Model 3: Fixed effects with entity and time fixed effects
    # res3 = single_regressor_entity_time_effects(data)
    # print(res3)
    # print(res3.params)
    # # Model 4: Multiple regressors with fixed effects
    # res4 = multi_regressor_entity_time_effects(data)
    # print(res4)
    # print(res4.params)


def _new_section(title: str):
    print("", "", "", sep="\n")
    print(Markdown(f"# {title}\n---"))
    print()


def add_lag(data: pd.DataFrame, periods=-1):
    possible_targets = [c for c in data.columns if c.startswith("rail_passengers")]
    shifted = data.shift(periods=periods)
    shifted.loc[:, possible_targets] = data.loc[:, possible_targets]
    return shifted.dropna()


def remove_accidents_outliers(data: pd.DataFrame):
    accident_cols = [c for c in data.columns if c.startswith("rail_accidents")]
    adjusted = data.copy()
    for col in accident_cols:
        adjusted.loc[
            data[col] < data[col].quantile(0.05),
            col,
        ] = None
        adjusted.loc[
            data[col] > data[col].quantile(0.95),
            col,
        ] = None
    return adjusted.dropna()


country_codes = {
    "Austria": "AT",
    "Belgium": "BE",
    "Bulgaria": "BG",
    "Czechia": "CZ",
    "Germany": "DE",
    "Denmark": "DK",
    "Estonia": "EE",
    "Greece": "GR",
    "Spain": "ES",
    "Finland": "FI",
    "France": "FR",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Lithuania": "LT",
    "Latvia": "LV",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Portugal": "PT",
    "Romania": "RO",
    "Sweden": "SE",
    "Slovenia": "SI",
    "Slovakia": "SK",
    "Switzerland": "CH",
}


@app.command()
def plot():
    geo_json_url = "https://raw.githubusercontent.com/leakyMirror/map-of-europe/refs/heads/master/GeoJSON/europe.geojson"
    europe = geopandas.read_file(geo_json_url).to_crs("EPSG:3035")
    countries_to_remove = ["RU", "TR", "GE", "AM", "AZ", "IL", "CY"]
    europe = europe[~europe["ISO2"].isin(countries_to_remove)]

    data = load_cached_data().reset_index()
    data["country"] = data["geo"]
    data = data.set_index("geo").rename(country_codes).reset_index()
    data["rail_pop"] = data["rail_passengers"] / data["population"]
    data["accidents_pkm"] = data["rail_accidents"] / data["rail_passengers"]
    merged = europe.merge(data, left_on="ISO2", right_on="geo", how="left")
    o_y = merged["year"]
    merged["year"] = o_y.fillna(2024)
    ax = merged[merged["year"] == 2024].plot(
        column="rail_pop",
        edgecolor="black",
        cmap=matplotlib.cm.Greens,
        missing_kwds={"color": "lightgrey"},
    )
    ax.axis("off")
    ax.set_title("Passenger-km by population (2024)")
    plt.show()
    merged["year"] = o_y.fillna(2024)
    ax = merged[merged["year"] == 2024].plot(
        column="cars_capita",
        edgecolor="black",
        cmap=matplotlib.cm.Greens,
        missing_kwds={"color": "lightgrey"},
    )
    ax.axis("off")
    ax.set_title("Cars pro capita (2024)")
    plt.show()
    merged["year"] = o_y.fillna(2024)
    ax = merged[merged["year"] == 2024].plot(
        column="rail_electrification_share",
        edgecolor="black",
        cmap=matplotlib.cm.Greens,
        missing_kwds={"color": "lightgrey"},
        legend=True,
    )
    ax.set_title("Rail electrification share (2024)")
    ax.axis("off")
    plt.show()
    merged["year"] = o_y.fillna(2024)
    ax = merged[merged["year"] == 2024].plot(
        column="accidents_pkm",
        edgecolor="black",
        cmap=matplotlib.cm.Reds,
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Rail accidents by passenger-km (2024)")
    ax.axis("off")
    plt.show()
    merged["year"] = o_y.fillna(2008)
    ax = merged[merged["year"] == 2008].plot(
        column="accidents_pkm",
        edgecolor="black",
        cmap=matplotlib.cm.Reds,
        missing_kwds={"color": "lightgrey"},
    )
    ax.set_title("Rail accidents by passenger-km (2008)")
    ax.axis("off")
    plt.show()


@app.command
def corr():
    data = load_cached_data()
    predictors = [
        "rail_accidents_div_pkm_log",
        "rail_electrification_share",
        "cars_capita_log",
        "gdp_capita_log",
        "railway_density_log",
    ]
    variables = ["rail_passengers_pop_log", *predictors]
    corr = data[variables].corr("kendall")
    mask = triu(ones_like(corr, dtype=bool))

    seaborn.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(20, 15))
    cmap = seaborn.diverging_palette(230, 20, as_cmap=True)
    seaborn.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        ax=ax,
    )
    ax.margins(x=0.5, y=0.5)
    plt.show()
    fig.savefig("correlation_matrix.png", bbox_inches="tight")


app()
