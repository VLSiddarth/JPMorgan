"""
ECB Statistical Data Warehouse (SDW) Connector
Access European economic data directly from ECB

Features:
- NO API KEY REQUIRED (completely free!)
- Euro area GDP, inflation, unemployment
- Bank lending data (credit impulse)
- ECB policy rates
- Financial stability indicators
- M3 money supply

API Documentation: https://data.ecb.europa.eu/help/api/overview

Author: JPMorgan Dashboard Team
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union
from datetime import datetime, timedelta
from pathlib import Path
import time
import json
import logging
from dataclasses import dataclass
from enum import Enum

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from config.settings import settings

logger = logging.getLogger(__name__)


class ECBFrequency(Enum):
    """Data frequency options"""
    ANNUAL = 'A'
    QUARTERLY = 'Q'
    MONTHLY = 'M'
    WEEKLY = 'W'
    DAILY = 'D'


class ECBDataset(Enum):
    """ECB dataset identifiers"""
    BSI = 'BSI'  # Balance Sheet Items (bank lending)
    ICP = 'ICP'  # Inflation (HICP)
    MNA = 'MNA'  # National Accounts (GDP)
    RTD = 'RTD'  # Retail Trade
    STS = 'STS'  # Short-term statistics
    FM = 'FM'    # Financial Markets


@dataclass
class ECBSeries:
    """ECB series configuration"""
    flow_ref: str
    name: str
    description: str
    frequency: ECBFrequency
    dataset: ECBDataset
    unit: str
    
    def get_key(self) -> str:
        """Get series key for API"""
        return f"{self.dataset.value}.{self.frequency.value}.{self.flow_ref}"


class ECBSDWConnector:
    """
    European Central Bank Statistical Data Warehouse Connector
    
    Access official European economic statistics:
    - GDP, inflation, unemployment
    - Bank lending and credit growth
    - Money supply (M1, M2, M3)
    - ECB policy rates
    - Financial stability indicators
    
    No API key required - completely FREE!
    
    Example:
        >>> ecb = ECBSDWConnector()
        >>> gdp = ecb.get_gdp_growth()
        >>> credit = ecb.get_credit_impulse()
        >>> hicp = ecb.get_hicp_inflation()
    """
    
    BASE_URL = "https://data-api.ecb.europa.eu/service"
    
    # Pre-defined series
    SERIES = {
        # GDP
        'EA_GDP': ECBSeries(
            'B1GQ.EUR.V.N',
            'Euro Area GDP',
            'Gross Domestic Product at market prices',
            ECBFrequency.QUARTERLY,
            ECBDataset.MNA,
            'millions_eur'
        ),
        
        # Inflation (HICP)
        'EA_HICP': ECBSeries(
            'N.U2.Y.000000.3.INX',
            'Euro Area HICP',
            'Harmonized Index of Consumer Prices',
            ECBFrequency.MONTHLY,
            ECBDataset.ICP,
            'index_2015=100'
        ),
        
        # Unemployment
        'EA_UNEMPLOYMENT': ECBSeries(
            'NSA.RTT000.4F',
            'Euro Area Unemployment',
            'Harmonized unemployment rate',
            ECBFrequency.MONTHLY,
            ECBDataset.STS,
            'percent'
        ),
        
        # Bank Lending
        'EA_LOANS_NFC': ECBSeries(
            'M.U2.N.A.A20.A.1.U2.2250.Z01.E',
            'Loans to Non-Financial Corporations',
            'Outstanding amounts of loans to NFCs',
            ECBFrequency.MONTHLY,
            ECBDataset.BSI,
            'millions_eur'
        ),
        
        'EA_LOANS_HOUSEHOLDS': ECBSeries(
            'M.U2.N.A.A20.A.1.U2.2240.Z01.E',
            'Loans to Households',
            'Outstanding amounts of loans to households',
            ECBFrequency.MONTHLY,
            ECBDataset.BSI,
            'millions_eur'
        ),
        
        # Money Supply
        'EA_M3': ECBSeries(
            'M.U2.N.A.M30.X.1.Z01.Z01.Z',
            'M3 Money Supply',
            'Broad money aggregate M3',
            ECBFrequency.MONTHLY,
            ECBDataset.BSI,
            'millions_eur'
        ),
    }
    
    def __init__(self, cache_dir: Optional[Path] = None, timeout: int = 30):
        """
        Initialize ECB SDW connector
        
        Args:
            cache_dir: Directory for caching responses
            timeout: Request timeout in seconds
        """
        self.cache_dir = cache_dir or settings.CACHE_DIR / 'ecb'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/vnd.sdmx.data+json;version=1.0.0'
        })
        
        logger.info("✅ ECB SDW Connector initialized (no API key needed!)")
    
    def _get_cache_path(self, series_key: str) -> Path:
        """Get cache file path"""
        safe_key = series_key.replace('/', '_').replace('.', '_')
        return self.cache_dir / f"{safe_key}.json"
    
    def _load_from_cache(self, series_key: str, max_age_hours: int = 24) -> Optional[pd.Series]:
        """Load from cache if recent"""
        cache_path = self._get_cache_path(series_key)
        
        if not cache_path.exists():
            return None
        
        try:
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > max_age_hours * 3600:
                return None
            
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            series = pd.Series(
                data['values'],
                index=pd.to_datetime(data['dates'])
            )
            
            logger.debug(f"📂 Loaded {series_key} from cache")
            return series
            
        except Exception as e:
            logger.warning(f"Cache load failed for {series_key}: {e}")
            return None
    
    def _save_to_cache(self, series_key: str, data: pd.Series):
        """Save to cache"""
        cache_path = self._get_cache_path(series_key)
        
        try:
            cache_data = {
                'series_key': series_key,
                'dates': data.index.strftime('%Y-%m-%d').tolist(),
                'values': data.tolist(),
                'cached_at': datetime.now().isoformat()
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            logger.debug(f"💾 Cached {series_key}")
            
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
    
    def _fetch_data(
        self,
        flow_ref: str,
        key: str = '',
        start_period: Optional[str] = None,
        end_period: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Fetch data from ECB API
        
        Args:
            flow_ref: Data flow reference (e.g., 'MNA')
            key: Series key
            start_period: Start period (YYYY-MM-DD)
            end_period: End period (YYYY-MM-DD)
            
        Returns:
            Raw JSON response
        """
        
        # Build URL
        url = f"{self.BASE_URL}/data/{flow_ref}/{key}"
        
        params = {}
        if start_period:
            params['startPeriod'] = start_period
        if end_period:
            params['endPeriod'] = end_period
        
        try:
            logger.debug(f"Fetching ECB data: {url}")
            response = self.session.get(
                url,
                params=params,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                logger.warning(f"Series not found: {flow_ref}/{key}")
                return None
            else:
                logger.error(f"ECB API error {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {flow_ref}")
            return None
        except Exception as e:
            logger.error(f"Error fetching ECB data: {e}")
            return None
    
    def _parse_response(self, response: Dict) -> Optional[pd.Series]:
        """
        Parse ECB JSON response to pandas Series
        
        Args:
            response: Raw JSON response
            
        Returns:
            pandas.Series with datetime index
        """
        
        try:
            # Navigate JSON structure
            datasets = response.get('data', {}).get('dataSets', [])
            if not datasets:
                logger.warning("No datasets in response")
                return None
            
            dataset = datasets[0]
            series_data = dataset.get('series', {})
            
            if not series_data:
                logger.warning("No series data in response")
                return None
            
            # Get first series (usually only one)
            series_key = list(series_data.keys())[0]
            observations = series_data[series_key].get('observations', {})
            
            if not observations:
                logger.warning("No observations in series")
                return None
            
            # Extract dates and values
            structure = response.get('data', {}).get('structure', {})
            dimensions = structure.get('dimensions', {}).get('observation', [])
            
            # Find time dimension
            time_dim = None
            for dim in dimensions:
                if dim.get('id') == 'TIME_PERIOD':
                    time_dim = dim
                    break
            
            if not time_dim:
                logger.error("TIME_PERIOD dimension not found")
                return None
            
            time_values = time_dim.get('values', [])
            
            # Build series
            dates = []
            values = []
            
            for idx, obs in observations.items():
                time_idx = int(idx)
                if time_idx < len(time_values):
                    date_str = time_values[time_idx].get('id')
                    value = obs[0] if isinstance(obs, list) else obs
                    
                    if date_str and value is not None:
                        dates.append(pd.to_datetime(date_str))
                        values.append(float(value))
            
            if not dates:
                logger.warning("No valid data points")
                return None
            
            series = pd.Series(values, index=dates).sort_index()
            
            logger.info(f"✅ Parsed ECB data: {len(series)} observations")
            return series
            
        except Exception as e:
            logger.error(f"Error parsing ECB response: {e}")
            return None
    
    def get_series(
        self,
        series_name: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        use_cache: bool = True
    ) -> Optional[pd.Series]:
        """
        Get ECB time series
        
        Args:
            series_name: Series name from SERIES dict (e.g., 'EA_GDP')
            start_date: Start date
            end_date: End date
            use_cache: Use cache if available
            
        Returns:
            pandas.Series with data
            
        Example:
            >>> ecb = ECBSDWConnector()
            >>> gdp = ecb.get_series('EA_GDP', start_date='2020-01-01')
            >>> print(gdp.tail())
        """
        
        if series_name not in self.SERIES:
            logger.error(f"Unknown series: {series_name}")
            return None
        
        series_config = self.SERIES[series_name]
        series_key = series_config.get_key()
        
        # Check cache
        if use_cache:
            cached = self._load_from_cache(series_key)
            if cached is not None:
                return cached
        
        # Format dates
        start_period = None
        end_period = None
        
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            start_period = start_date.strftime('%Y-%m-%d')
        
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            end_period = end_date.strftime('%Y-%m-%d')
        
        # Fetch data
        response = self._fetch_data(
            series_config.dataset.value,
            series_config.flow_ref,
            start_period,
            end_period
        )
        
        if not response:
            return None
        
        # Parse data
        data = self._parse_response(response)
        
        if data is not None and use_cache:
            self._save_to_cache(series_key, data)
        
        return data
    
    def get_gdp_growth(
        self,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """
        Get Euro Area GDP growth (year-over-year %)
        
        Returns:
            GDP growth rate series
        """
        
        gdp = self.get_series('EA_GDP', start_date=start_date)
        
        if gdp is None:
            return None
        
        # Calculate YoY growth
        gdp_growth = gdp.pct_change(periods=4) * 100  # Quarterly data, 4 periods = 1 year
        
        return gdp_growth.dropna()
    
    def get_hicp_inflation(
        self,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """
        Get HICP inflation (year-over-year %)
        
        Returns:
            Inflation rate series
        """
        
        hicp = self.get_series('EA_HICP', start_date=start_date)
        
        if hicp is None:
            return None
        
        # Calculate YoY inflation
        inflation = hicp.pct_change(periods=12) * 100  # Monthly data, 12 periods = 1 year
        
        return inflation.dropna()
    
    def get_credit_impulse(
        self,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """
        Calculate credit impulse (YoY change in credit growth)
        
        Credit impulse = change in credit flow as % of GDP
        Proxy: YoY % change in total loans
        
        Returns:
            Credit impulse series
        """
        
        # Get loan data
        loans_nfc = self.get_series('EA_LOANS_NFC', start_date=start_date)
        loans_hh = self.get_series('EA_LOANS_HOUSEHOLDS', start_date=start_date)
        
        if loans_nfc is None or loans_hh is None:
            logger.warning("Could not fetch loan data for credit impulse")
            return None
        
        # Total loans
        total_loans = loans_nfc + loans_hh
        
        # Calculate YoY growth
        credit_growth = total_loans.pct_change(periods=12) * 100
        
        # Credit impulse = change in growth rate
        credit_impulse = credit_growth.diff()
        
        return credit_impulse.dropna()
    
    def get_m3_growth(
        self,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.Series]:
        """
        Get M3 money supply growth (YoY %)
        
        Returns:
            M3 growth series
        """
        
        m3 = self.get_series('EA_M3', start_date=start_date)
        
        if m3 is None:
            return None
        
        # Calculate YoY growth
        m3_growth = m3.pct_change(periods=12) * 100
        
        return m3_growth.dropna()
    
    def get_macro_snapshot(self) -> Dict[str, float]:
        """
        Get latest macro indicators
        
        Returns:
            Dictionary with latest values
        """
        
        snapshot = {}
        
        # GDP growth
        gdp_growth = self.get_gdp_growth()
        if gdp_growth is not None and len(gdp_growth) > 0:
            snapshot['gdp_growth_yoy'] = float(gdp_growth.iloc[-1])
        
        # Inflation
        inflation = self.get_hicp_inflation()
        if inflation is not None and len(inflation) > 0:
            snapshot['hicp_inflation_yoy'] = float(inflation.iloc[-1])
        
        # Credit impulse
        credit = self.get_credit_impulse()
        if credit is not None and len(credit) > 0:
            snapshot['credit_impulse'] = float(credit.iloc[-1])
        
        # M3 growth
        m3 = self.get_m3_growth()
        if m3 is not None and len(m3) > 0:
            snapshot['m3_growth_yoy'] = float(m3.iloc[-1])
        
        return snapshot
    
    def health_check(self) -> Dict[str, any]:
        """
        Check ECB API health
        
        Returns:
            Health status dictionary
        """
        
        status = {
            'ecb_api': 'unknown',
            'cache_dir': str(self.cache_dir),
            'cached_series': len(list(self.cache_dir.glob('*.json'))),
            'test_query': False
        }
        
        try:
            # Test with simple query
            test_data = self._fetch_data('ICP', 'M.U2.N.000000.4.ANR')
            
            if test_data:
                status['ecb_api'] = 'healthy'
                status['test_query'] = True
            else:
                status['ecb_api'] = 'no_data'
                
        except Exception as e:
            status['ecb_api'] = f'error: {str(e)}'
        
        return status


if __name__ == "__main__":
    # Test ECB SDW connector
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("TESTING ECB SDW CONNECTOR")
    print("="*70)
    
    ecb = ECBSDWConnector()
    
    # Test 1: Health check
    print("\n[Test 1] Health Check:")
    health = ecb.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # Test 2: GDP growth
    print("\n[Test 2] Euro Area GDP Growth:")
    gdp = ecb.get_gdp_growth(start_date=datetime(2023, 1, 1))
    if gdp is not None and len(gdp) > 0:
        print(f"  Latest GDP growth: {gdp.iloc[-1]:.2f}%")
        print(f"  Data points: {len(gdp)}")
    
    # Test 3: Inflation
    print("\n[Test 3] HICP Inflation:")
    inflation = ecb.get_hicp_inflation(start_date=datetime(2023, 1, 1))
    if inflation is not None and len(inflation) > 0:
        print(f"  Latest inflation: {inflation.iloc[-1]:.2f}%")
    
    # Test 4: Credit impulse
    print("\n[Test 4] Credit Impulse:")
    credit = ecb.get_credit_impulse(start_date=datetime(2022, 1, 1))
    if credit is not None and len(credit) > 0:
        print(f"  Latest credit impulse: {credit.iloc[-1]:.2f}%")
    
    # Test 5: Macro snapshot
    print("\n[Test 5] Macro Snapshot:")
    snapshot = ecb.get_macro_snapshot()
    for key, value in snapshot.items():
        print(f"  {key}: {value:.2f}")
    
    print("\n" + "="*70)
    print("✅ ECB SDW CONNECTOR TEST COMPLETE")
    print("="*70)