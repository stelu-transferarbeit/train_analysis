from pathlib import Path

import pandas as pd
from cyclopts import App

app = App()


datasets = {
    "rail_length": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/rail_if_line_na/1.0/*.*.*.*.*?c[freq]=A&c[unit]=KM&c[tra_infr]=TOTAL,RL_ELC&c[tra_meas]=FR_ONL,TOTAL&c[TIME_PERIOD]=2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL",
    "cars_capita": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/road_eqs_carhab/1.0/*.*.*?c[freq]=A&c[unit]=NR&c[geo]=EU27_2020,BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,IS,LI,NO,CH,UK,BA,ME,MD,MK,GE,AL,RS,TR,UA,XK&c[TIME_PERIOD]=2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL",
    "rail_passengers": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/rail_pa_total/1.0/*.*.*?c[freq]=A&c[unit]=MIO_PKM&c[TIME_PERIOD]=2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL",
    # "population": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0/*.*.*.*.*?c[freq]=A&c[unit]=NR&c[age]=TOTAL&c[sex]=T&c[geo]=BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,FX,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,IS,LI,NO,CH,UK,BA,ME,MD,MK,GE,AL,RS,TR,UA,XK,AD,BY,MC,RU,SM,AM,AZ&c[TIME_PERIOD]=2025,2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only",
    "gdp_per_capita": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/sdg_08_10/1.0/*.*.*.*?c[freq]=A&c[unit]=CLV20_EUR_HAB&c[na_item]=B1GQ&c[geo]=BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,CY,LV,LT,LU,HU,MT,NL,AT,PL,PT,RO,SI,SK,FI,SE,IS,NO,CH,AL,BA,EA19,ME,MK,RS,TR&c[TIME_PERIOD]=2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL",
    # "area": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_r_d3area/1.0/*.*.*.*?c[freq]=A&c[unit]=KM2&c[landuse]=TOTAL&c[geo]=BE,BE1,BE10,BE100,BE2,BE21,BE211,BE212,BE213,BE22,BE221,BE222,BE223,BE23,BE231,BE232,BE233,BE234,BE235,BE236,BE24,BE241,BE242,BE25,BE251,BE252,BE253,BE254,BE255,BE256,BE257,BE258,BE3,BE31,BE310,BE32,BE321,BE322,BE323,BE324,BE325,BE326,BE327,BE33,BE331,BE332,BE334,BE335,BE336,BE34,BE341,BE342,BE343,BE344,BE345,BE35,BE351,BE352,BE353,BG,BG3,BG31,BG311,BG312,BG313,BG314,BG315,BG32,BG321,BG322,BG323,BG324,BG325,BG33,BG331,BG332,BG333,BG334,BG34,BG341,BG342,BG343,BG344,BG4,BG41,BG411,BG412,BG413,BG414,BG415,BG42,BG421,BG422,BG423,BG424,BG425,CZ,CZ0,CZ01,CZ010,CZ02,CZ020,CZ03,CZ031,CZ032,CZ04,CZ041,CZ042,CZ05,CZ051,CZ052,CZ053,CZ06,CZ063,CZ064,CZ07,CZ071,CZ072,CZ08,CZ080,DK,DK0,DK01,DK011,DK012,DK013,DK014,DK02,DK021,DK022,DK03,DK031,DK032,DK04,DK041,DK042,DK05,DK050,DE,IT,ITC,ITC1,ITC11,ITC12,ITC13,ITC14,ITC15,ITC16,ITC17,ITC18,ITC2,ITC20,ITC3,ITC31,ITC32,ITC33,ITC34,ITC4,ITF,ITG,ITH,ITI,FR,FR1,FR10,FR2,FR21,FR22,FR23,FR24,FR25,FR26,FR3,FR30,FR4,FR41,FR42,FR43,FR5,FR51,FR52,FR53,FR6,FR61,FR62,FR63,FR7,FR71,FR72,FR8,FR81,FR82,FR83,ES,ES1,ES11,ES12,ES13,ES2,ES21,ES22,ES23,ES24,ES3,ES30,ES4,ES41,ES42,ES43,ES5,ES51,ES52,ES53,ES6,ES61,ES62,ES63,ES64,ES7,ES70,NL,NL1,NL2,NL3,NL4,AT,AT1,AT2,AT3,PL,PL1,PL2,PL3,PL4,PL5,PL6,PT,PT1,PT15,PT16,PT17,PT18,PT2,PT20,PT3,PT30,RO,RO1,RO2,RO3,RO4,SE,SE1,SE2,SE3,FI,FI1,FI2,FI20,EE,IE,EL,HR,CY,LV,LT,LU,HU,MT,SI,SK,IS,LI,NO,CH,UK,ME,MK,AL,TR&c[TIME_PERIOD]=2015&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only",
    "modal_split": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/tran_hv_ms_psmod/1.0?compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&c[TIME_PERIOD]=2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008&c[vehicle]=TRN,CAR,BUS_TOT,AC&returnData=ALL",
    # "rail_investment": "https://sdmx.oecd.org/public/rest/data/OECD.ITF,DSD_INFRINV@DF_INFRINV,1.0/.A..EUR.TOT_INL+MAR+AIR.Q",
    "rail_accidents": "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/tran_sf_railac/1.0/*.*.*.*?c[freq]=A&c[unit]=NR&c[accident]=TOTAL&c[geo]=EU27_2020,BE,BG,CZ,DK,DE,EE,IE,EL,ES,FR,HR,IT,LV,LT,LU,HU,NL,AT,PL,PT,RO,SI,SK,FI,SE,CHUNNEL,NO,CH,UK,ME,MK,TR&c[TIME_PERIOD]=2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024&compress=true&format=csvdata&formatVersion=1.0&lang=en&labels=label_only&returnData=ALL",
}


def transform_rail_length(data: pd.DataFrame):
    # Correct german data for 2001
    pass_data = data
    pass_data.loc[
        (pass_data["geo"] == "Germany")
        & (pass_data["TIME_PERIOD"] == 2001)
        & (pass_data["tra_meas"] != "Total")
        & (pass_data["tra_infr"] == "Total"),
        "OBS_VALUE",
    ] /= 1000
    print(
        pass_data.loc[
            (pass_data["geo"] == "Germany")
            & (pass_data["TIME_PERIOD"] == 2001)
            & (pass_data["tra_meas"] != "Total")
            & (pass_data["tra_infr"] == "Total"),
            "OBS_VALUE",
        ]
    )
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
            "OBS_VALUE_electrified": "rail_electrification_quota",
        },
        axis="columns",
    ).loc[
        :,
        ~pass_data.columns.isin(["tra_infr_electrified", "tra_infr"]),
    ]


def transform_modal_split(data: pd.DataFrame):
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


def load_data():
    data = None
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
    return data


@app.default
def analyze():
    download()
    data = load_data()
    print(data)
    print("Proceeding with analysis")


app()
