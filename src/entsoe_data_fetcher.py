"""ENTSO-E API Data Fetcher - Hungary Load Data (15-minute resolution)

This script downloads historical electricity load data for Hungary from the ENTSO-E API.
It fetches data month-by-month, converts it into a clean 15-minute UTC time series,
fills missing timestamps using linear interpolation, and saves results as CSV.

Main goal: get a continuous, model-ready load dataset for 2015-2024."""

import requests
import pandas as pd
import xmltodict
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import logging
import time
from calendar import monthrange
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ENTSOEDataFetcher:
    """ENTSOE data fetchici - Hungary için 15 dakikalık load datasi"""
    
    def __init__(
        self,
        api_key: str = "f3b6a3a9-1cef-4f25-83cf-1045c82019e5",
        base_url: str = "https://web-api.tp.entsoe.eu/api",
        bidding_zone: str = "10YHU-MAVIR----U",
        timezone: str = "Europe/Budapest"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.bidding_zone = bidding_zone
        self.local_tz = pytz.timezone(timezone)
        self.utc_tz = pytz.UTC
        
        # Retry mekanizması
        self.session = requests.Session()
        retry = Retry(connect=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
    
    def fetch_month(self, year: int, month: int) -> pd.DataFrame:
        """Bir ayın datasini fetch"""
        last_day = monthrange(year, month)[1]
        
        # periodEnd: Bir sonraki günün başlangıcını kullan (son aralıkların fetchilmesi için)
        # Örnek: Aralık 2024 için 2025-01-01 00:00 kullanılır
        end_date = datetime(year, month, last_day) + timedelta(days=1)
        
        # API formatı: YYYYMMDDHHMM
        p_start = f"{year}{month:02d}010000"
        p_end = f"{end_date.year}{end_date.month:02d}{end_date.day:02d}0000"
        
        params = {
            "securityToken": self.api_key,
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": self.bidding_zone,
            "periodStart": p_start,
            "periodEnd": p_end
        }
        
        all_data = []
        
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data_dict = xmltodict.parse(response.content)
                
                if 'GL_MarketDocument' in data_dict:
                    # TimeSeries listesi
                    ts_list = data_dict['GL_MarketDocument'].get('TimeSeries', [])
                    if not isinstance(ts_list, list):
                        ts_list = [ts_list]
                    
                    for ts in ts_list:
                        # Period listesi (liste olabilir!)
                        periods = ts.get('Period', [])
                        if not isinstance(periods, list):
                            periods = [periods]
                        
                        for period in periods:
                            start_str = period['timeInterval']['start']
                            start_dt = pd.to_datetime(start_str).tz_convert('UTC')
                            resolution = period['resolution']
                            
                            # Çözünürlük
                            if resolution == 'PT15M':
                                step = timedelta(minutes=15)
                            elif resolution == 'PT60M':
                                step = timedelta(hours=1)
                            else:
                                step = timedelta(minutes=15)
                            
                            # Point listesi (liste olabilir!)
                            points = period.get('Point', [])
                            if not isinstance(points, list):
                                points = [points]
                            
                            for p in points:
                                pos = int(p['position'])
                                qty = float(p['quantity'])
                                curr_time = start_dt + (pos - 1) * step
                                
                                all_data.append({
                                    'datetime': curr_time,
                                    'load_MW': qty
                                })
                
                if all_data:
                    logger.info(f"✓ {year}-{month:02d}: {len(all_data)} rows")
                else:
                    logger.warning(f"⚠ {year}-{month:02d}: Veri yok")
            else:
                logger.error(f"❌ {year}-{month:02d}: HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"❌ {year}-{month:02d}: {e}")
        
        if not all_data:
            return pd.DataFrame(columns=['datetime', 'load_MW'])
        
        df = pd.DataFrame(all_data)
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        
        return df
    
    def fetch_range(self, start_date: datetime, end_date: datetime, show_progress: bool = True) -> pd.DataFrame:
        """Date rangenı ay ay fetch"""
        # Tarihleri timezone-aware hale getir (local timezone)
        if start_date.tzinfo is None:
            start_date = self.local_tz.localize(start_date)
        else:
            start_date = start_date.astimezone(self.local_tz)
        
        if end_date.tzinfo is None:
            end_date = self.local_tz.localize(end_date)
        else:
            end_date = end_date.astimezone(self.local_tz)
        
        all_data = []
        
        # Ay listesi generate
        months = []
        temp = start_date
        while temp <= end_date:
            months.append((temp.year, temp.month))
            if temp.month == 12:
                temp = self.local_tz.localize(datetime(temp.year + 1, 1, 1))
            else:
                temp = self.local_tz.localize(datetime(temp.year, temp.month + 1, 1))
        
        if show_progress:
            print(f"\n📅 {len(months)} ay için data fetchiliyor...\n")
        
        for idx, (year, month) in enumerate(months, 1):
            if show_progress:
                print(f"[{idx}/{len(months)}] {year}-{month:02d}...", end=" ", flush=True)
            
            month_data = self.fetch_month(year, month)
            
            if not month_data.empty:
                all_data.append(month_data)
                if show_progress:
                    print(f"✓ {len(month_data)} rows")
            else:
                if show_progress:
                    print("⚠ Veri yok")
            
            if month % 3 == 0 and show_progress:
                print()  # Satır atla
            
            time.sleep(0.3)  # Rate limiting
        
        if not all_data:
            return pd.DataFrame(columns=['datetime', 'load_MW'])
        
        # Birleştir
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
        
        if show_progress:
            print(f"\n✅ Toplam {len(df):,} rows fetchildi")
            print(f"   Date range: {df['datetime'].min()} - {df['datetime'].max()}\n")
        
        return df
    
    def reindex_and_fill(self, df: pd.DataFrame) -> tuple:
        """
        Veriyi mükemmel zaman ızgarasına oturt, missingleri tespit et ve interpolation yap.
        Çekilen datanin gerfetch tarih aralığına göre çalışır.
        
        Args:
            df: Veri DataFrame'i
        
        Returns:
            (df_filled, df_report) - Doldurulmuş data ve missing data report
        """
        # Duplicate'leri temizle ve indexle
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'], keep='last')
        df = df.set_index('datetime')
        
        # Empty data kontrolü
        if df.empty:
            return df, pd.DataFrame(columns=['Start', 'End', 'Sure_Hours', 'Num_Points'])
        
        # Çekilen datanin gerfetch tarih aralığını kullan
        min_date = df.index.min()
        max_date = df.index.max()
        
        # Start: İlk datanin olduğu günün başlangıcı (00:00)
        start_date = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
        # End: Son datanin olduğu günün sonu (23:45)
        end_date = max_date.replace(hour=23, minute=45, second=0, microsecond=0)
        
        # Mükemmel zaman ızgarası generate
        full_idx = pd.date_range(
            start=start_date,
            end=end_date,
            freq='15min',
            tz='UTC'
        )
        
        # Veriyi bu ızgaraya oturt
        df_full = df.reindex(full_idx)
        df_full.index.name = 'datetime'
        
        # Eksik dataleri tespit et
        missing_mask = df_full['load_MW'].isna()
        missing_count = missing_mask.sum()
        
        report_list = []
        
        if missing_count > 0:
            # Ardışık boşlukları grupla
            groups = missing_mask.ne(missing_mask.shift()).cumsum()
            
            # Sadece NaN olan grupları al
            for _, group in df_full[missing_mask].groupby(groups):
                start_gap = group.index.min()
                end_gap = group.index.max()
                duration_hours = (group.index.size * 15) / 60
                
                report_list.append({
                    'Start': start_gap,
                    'End': end_gap,
                    'Sure_Hours': duration_hours,
                    'Num_Points': group.index.size
                })
        
        # Interpolasyon yap
        df_full['load_MW'] = df_full['load_MW'].interpolate(method='linear', limit_direction='both')
        
        df_report = pd.DataFrame(report_list)
        
        return df_full, df_report
    
    def save_csv(self, df: pd.DataFrame, filepath: str):
        """CSV'ye save"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=True)
        logger.info(f"✅ Veri saved: {filepath}")


if __name__ == "__main__":
    fetcher = ENTSOEDataFetcher()
    
    start = datetime(2015, 1, 1)
    end = datetime(2024, 12, 31)
    
    # Veriyi fetch
    df_raw = fetcher.fetch_range(start, end)
    
    # Reindex ve interpolation (fetchilen datanin gerfetch tarih aralığına göre)
    df_filled, df_report = fetcher.reindex_and_fill(df_raw)
    
    # Kaydet
    fetcher.save_csv(df_filled, "data/raw/hungary_load_data_2015_2024.csv")
    
    if not df_report.empty:
        fetcher.save_csv(df_report, "data/raw/interpolation_report.csv")
        print(f"\n⚠️ {df_report['Num_Points'].sum()} data noktası interpolation ile dolduruldu")

