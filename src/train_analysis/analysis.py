import pandas
from linearmodels import PanelOLS, PooledOLS
from linearmodels.panel.results import PanelResults

single_regressor = ["cars_capita"]
multi_regressor = [
    "rail_accidents",
    "rail_electrification_share",
    "cars_10k",
    "gdp",
    "total_rail_length",
]


def _train(
    data: pandas.DataFrame, predictors: list[str], effects: list[str] | None = None
):
    # for country, _g in data.reset_index().groupby("geo"):
    #     print(f"Data for {country}")
    #     print(
    #         data.loc[
    #             country,
    #             ["rail_passengers", *predictors],
    #         ]
    #     )
    if effects is None or len(effects) == 0:
        return _pooled(data, predictors)
    return _panel(data, predictors, effects)


def _pooled(data: pandas.DataFrame, predictors: list[str]) -> PanelResults:
    return PooledOLS.from_formula(
        f"rail_passengers ~ 1 + {' + '.join(predictors)}",
        data=data,
    ).fit(cov_type="clustered", cluster_entity=True)


def _panel(
    data: pandas.DataFrame, predictors: list[str], effects: list[str]
) -> PanelResults:
    return PanelOLS.from_formula(
        f"rail_passengers ~ 1 + {' + '.join([*predictors, *effects])}",
        data=data,
    ).fit(cov_type="clustered", cluster_entity=True)


def single_regressor_no_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor)


def single_regressor_entity_fixed_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor, ["EntityEffects"])


def single_regressor_entity_time_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor, ["EntityEffects", "TimeEffects"])


def multi_regressor_entity_time_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, multi_regressor, ["EntityEffects", "TimeEffects"])
