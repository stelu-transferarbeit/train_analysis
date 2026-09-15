import pandas
from linearmodels import PanelOLS, PooledOLS, RandomEffects
from linearmodels.panel.results import PanelResults
from statsmodels.api import add_constant

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
        return pooled(data, predictors)
    return panel(data, predictors, effects)


def pooled(data: pandas.DataFrame, target: str, predictors: list[str]) -> PanelResults:
    predict_data = add_constant(data[predictors])
    model = PooledOLS(data[target], predict_data).fit()
    return model


def panel(
    data: pandas.DataFrame,
    target: str,
    predictors: list[str],
    entity_effects: bool = False,
    time_effects: bool = False,
) -> PanelResults:
    predict_data = add_constant(data[predictors])
    model = PanelOLS(
        data[target],
        predict_data,
        entity_effects=entity_effects,
        time_effects=time_effects,
    ).fit()
    return model


def random_effects(
    data: pandas.DataFrame, target: str, predictors: list[str]
) -> PanelResults:
    predict_data = add_constant(data[predictors])
    model = RandomEffects(data[target], predict_data).fit()
    return model


def single_regressor_no_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor)


def multiple_regressor_no_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, multi_regressor)


def single_regressor_entity_fixed_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor, ["EntityEffects"])


def single_regressor_entity_time_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, single_regressor, ["EntityEffects", "TimeEffects"])


def multi_regressor_entity_time_effects(data: pandas.DataFrame) -> PanelResults:
    return _train(data, multi_regressor, ["EntityEffects", "TimeEffects"])
