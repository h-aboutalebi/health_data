import logging

logger = logging.getLogger(__name__)
from xgboost import XGBRegressor
from engine.lightgbm_wrapper import LightgbmWrapper

def get_model_type(model_name):

    if (model_name == "XGBoost"):
        model = XGBRegressor()
        logger.info("XGBRegressor has been initilized.")
    elif (model_name == "lightgbm"):
        model = LightgbmWrapper()
        logger.info("lightgbm has been initilized.")
    else:
        logger.info("model has not been implemented error! Model should be one of: [XGBoost]")
        raise NotImplementedError
    return model
