"""
General Helper Utilities

Common helper functions used across the JPMorgan European Equity Dashboard:

- Safe dictionary / nested access
- Number & percentage formatting
- Currency formatting (EUR/USD/GBP)
- JSON (de)serialization helpers
- DataFrame utilities (safe merge, column checks)
- Misc utilities (chunking, flattening, etc.)

All helpers are intentionally lightweight and standard-library first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


# -----------------------------------------------------------------------------
# Safe dictionary / nested access
# -----------------------------------------------------------------------------

def get_safe(
    mapping: Mapping[K, V],
    key: K,
    default: Optional[V] = None,
    *,
    log_missing: bool = False,
) -> Optional[V]:
    """
    Safely get a value from a mapping with an optional default.

    Args:
        mapping: Source mapping.
        key: Key to look up.
        default: Default value if key missing.
        log_missing: If True, logs when the key is missing.

    Returns:
        Value or default.
    """
    if key in mapping:
        return mapping[key]
    if log_missing:
        logger.debug("Key '%s' missing in mapping, returning default=%s", key, default)
    return default


def get_nested(
    mapping: Mapping[str, Any],
    path: Union[str, Sequence[str]],
    default: Any = None,
    *,
    separator: str = ".",
) -> Any:
    """
    Safely navigate nested dictionaries using a path.

    Args:
        mapping: Nested mapping.
        path: Either "a.b.c" or sequence ["a", "b", "c"].
        default: Value if any level is missing.
        separator: Separator to split string path.

    Example:
        d = {"a": {"b": {"c": 1}}}
        get_nested(d, "a.b.c") -> 1
        get_nested(d, "a.x.c", default=0) -> 0
    """
    if isinstance(path, str):
        keys = path.split(separator)
    else:
        keys = list(path)

    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def set_nested(
    mapping: MutableMapping[str, Any],
    path: Union[str, Sequence[str]],
    value: Any,
    *,
    separator: str = ".",
) -> None:
    """
    Set a value in a nested dict structure, creating intermediate levels.

    Args:
        mapping: Dict to modify (in place).
        path: Either "a.b.c" or list of keys.
        value: Value to set.
        separator: Path separator.
    """
    if isinstance(path, str):
        keys = path.split(separator)
    else:
        keys = list(path)

    current: MutableMapping[str, Any] = mapping
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], MutableMapping):
            current[key] = {}
        current = current[key]  # type: ignore[assignment]
    current[keys[-1]] = value


# -----------------------------------------------------------------------------
# Formatting helpers (percent, currency, etc.)
# -----------------------------------------------------------------------------

def format_percentage(
    value: Optional[float],
    decimals: int = 2,
    *,
    default: str = "N/A",
    include_sign: bool = False,
) -> str:
    """
    Format a float as percentage string.

    Args:
        value: Number in decimal (0.05 -> 5%).
        decimals: Number of decimal places.
        default: Fallback string for None/NaN.
        include_sign: If True, always include + / -.

    Returns:
        Percentage string, e.g. "5.00%" or "+5.00%".
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default

    val = float(value) * 100.0
    q = Decimal(val).quantize(Decimal(f"1.{'0' * decimals}"), rounding=ROUND_HALF_UP)

    if include_sign:
        return f"{q:+.{decimals}f}%"
    return f"{q:.{decimals}f}%"


def format_number(
    value: Optional[float],
    decimals: int = 2,
    *,
    default: str = "N/A",
    thousand_sep: str = ",",
) -> str:
    """
    Format a float with given decimals and thousands separator.

    Args:
        value: Numeric value.
        decimals: Decimal places.
        default: Fallback for None/NaN.
        thousand_sep: Thousands separator.

    Returns:
        Formatted string.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default

    fmt = f"{{:,.{decimals}f}}"
    s = fmt.format(float(value))
    if thousand_sep != ",":
        s = s.replace(",", thousand_sep)
    return s


def format_currency(
    value: Optional[float],
    currency: str = "EUR",
    decimals: int = 2,
    *,
    default: str = "N/A",
) -> str:
    """
    Format a value as currency string (EUR / USD / GBP, etc.).

    Args:
        value: Amount.
        currency: Currency code, e.g. "EUR", "USD", "GBP".
        decimals: Decimal places.
        default: Fallback for None/NaN.

    Returns:
        Currency string.
    """
    symbol_map = {
        "EUR": "€",
        "USD": "$",
        "GBP": "£",
        "INR": "₹",
    }
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default

    symbol = symbol_map.get(currency.upper(), currency.upper() + " ")
    fmt = f"{{:,.{decimals}f}}"
    return f"{symbol}{fmt.format(float(value))}"


# -----------------------------------------------------------------------------
# JSON / serialization helpers
# -----------------------------------------------------------------------------

def to_json_safe(obj: Any) -> Any:
    """
    Convert an object into a JSON-serializable structure.

    Supports:
        - dataclasses
        - numpy scalars / arrays
        - pandas Series / DataFrame
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # Fallback to string representation
    return str(obj)


def dumps_json(obj: Any, **kwargs: Any) -> str:
    """
    Dump an object to a JSON string using `to_json_safe`.

    Args:
        obj: Object to serialize.
        kwargs: Extra args passed to json.dumps.

    Returns:
        JSON string.
    """
    return json.dumps(to_json_safe(obj), **kwargs)


def loads_json(s: str) -> Any:
    """
    Load JSON string into Python object.

    Thin wrapper around json.loads for consistency.
    """
    return json.loads(s)


# -----------------------------------------------------------------------------
# DataFrame helpers
# -----------------------------------------------------------------------------

def safe_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: Union[str, List[str]],
    how: str = "inner",
    *,
    suffixes: Tuple[str, str] = ("_x", "_y"),
    validate: Optional[str] = None,
) -> pd.DataFrame:
    """
    Safe wrapper around pandas.merge with logging.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        on: Column name(s) to merge on.
        how: Merge type (inner/left/right/outer).
        suffixes: Column suffixes.
        validate: Optional merge validation, e.g. 'one_to_many'.

    Returns:
        Merged DataFrame.
    """
    logger.debug(
        "Merging DataFrames: left=%s rows, right=%s rows, on=%s, how=%s",
        len(left),
        len(right),
        on,
        how,
    )
    try:
        merged = pd.merge(
            left,
            right,
            on=on,
            how=how,
            suffixes=suffixes,
            validate=validate,
        )
        logger.debug("Merge result: %s rows, %s columns", len(merged), len(merged.columns))
        return merged
    except Exception:
        logger.exception("DataFrame merge failed")
        raise


def ensure_columns(
    df: pd.DataFrame,
    required: Iterable[str],
    *,
    raise_error: bool = True,
) -> List[str]:
    """
    Ensure that a DataFrame contains required columns.

    Args:
        df: DataFrame to check.
        required: Iterable of required column names.
        raise_error: If True, raises ValueError on missing columns.

    Returns:
        List of missing columns (empty if none).
    """
    required_set = set(required)
    missing = [c for c in required_set if c not in df.columns]

    if missing:
        msg = f"DataFrame missing required columns: {missing}"
        if raise_error:
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return missing


def normalize_to_100(series: pd.Series) -> pd.Series:
    """
    Normalize a price/level series to a base of 100.

    Args:
        series: Input Series (index = dates, values = levels/prices).

    Returns:
        Series normalized so first non-NaN value is 100.
    """
    if series is None or series.empty:
        return series

    s = series.dropna()
    if s.empty:
        return series

    base = float(s.iloc[0])
    if base == 0:
        return series * 0.0

    return (series / base) * 100.0


# -----------------------------------------------------------------------------
# Misc helpers
# -----------------------------------------------------------------------------

def chunked(iterable: Iterable[T], size: int) -> Iterable[List[T]]:
    """
    Yield lists of `size` elements from an iterable.

    Args:
        iterable: Any iterable.
        size: Chunk size > 0.

    Yields:
        List[T] of length <= size.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    chunk: List[T] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def flatten(list_of_lists: Iterable[Iterable[T]]) -> List[T]:
    """
    Flatten a list of lists into a single list.

    Args:
        list_of_lists: e.g. [[1,2], [3], [4,5]]

    Returns:
        [1,2,3,4,5]
    """
    out: List[T] = []
    for sub in list_of_lists:
        out.extend(sub)
    return out


def safe_float(value: Any, default: float = float("nan")) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Any value.
        default: Value to return if conversion fails.

    Returns:
        float value or default.
    """
    try:
        if value is None:
            return default
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """
    Safely convert a value to int.

    Args:
        value: Any value.
        default: Value to return if conversion fails.

    Returns:
        int value or default.
    """
    try:
        if value is None:
            return default
        if isinstance(value, (np.floating, np.integer)):
            return int(value)
        return int(str(value).replace(",", ""))
    except Exception:
        return default


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    d = {"a": {"b": {"c": 1}}}
    print("get_nested a.b.c =", get_nested(d, "a.b.c"))
    print("get_nested a.x.c =", get_nested(d, "a.x.c", default="missing"))

    print("format_percentage(0.1234) =", format_percentage(0.1234))
    print("format_currency(1234.56, 'EUR') =", format_currency(1234.56, "EUR"))

    series = pd.Series([100, 105, 110], index=pd.date_range("2024-01-01", periods=3))
    print("normalize_to_100:", normalize_to_100(series).tolist())

    print("chunked 1..10:", list(chunked(range(1, 11), 4)))
