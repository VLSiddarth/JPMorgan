from src.data.processors.cleaner import cleaner
from src.data.processors.normalizer import normalizer, NormalizationConfig

# 1) clean raw prices
cleaned, clean_report = cleaner.clean_price_series(price_series, symbol="^STOXX50E")

# 2) normalize to z-scored daily returns
norm_cfg = NormalizationConfig(
    price_to_return=True,
    log_return=False,
    standardize="zscore",
    center=False,
    scale_by_vol=False,
)
normalized, norm_report = normalizer.normalize_series(
    cleaned,
    symbol="^STOXX50E",
    config=norm_cfg,
)
