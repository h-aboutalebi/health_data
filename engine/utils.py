import logging

logger = logging.getLogger(__name__)
from xgboost import XGBRegressor


def get_model_type(model_name):

    if (model_name == "XGBoost"):
        model = XGBRegressor()
        logger.info("XGBRegressor has been initilized!")
    else:
        logger.info("model has not been implemented error! Model should be one of: [XGBoost]")
        raise NotImplementedError
    return model
