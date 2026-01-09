"""Calendar Generator - Hungary (15-minute resolution)

This script builds detailed calendar features for electricity demand forecasting in Hungary.
The final output is in UTC with 15-minute steps. We still use local time (Europe/Budapest)
internally so we can correctly detect holidays, business hours, and DST.

In short: we convert time information into machine-learning friendly features (cyclical,
binary flags, and one-hot encodings)."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import logging
import holidays

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HungaryCalendarGenerator:
    """Hungary için detaylı calendar özellikleri generateucu"""
    
    def __init__(self, timezone: str = "Europe/Budapest"):
        self.local_tz = pytz.timezone(timezone)
        self.utc_tz = pytz.UTC
        
        # Hungary tatil günleri (holidays kütüphanesi)
        self.hungary_holidays = holidays.Hungary(years=range(2015, 2026))
        
        # İş saatleri (Hungary local time)
        self.business_hours_start = 8  # 08:00
        self.business_hours_end = 18   # 18:00
    
    def _is_school_holiday(self, date: datetime.date) -> bool:
        """
        Okul tatili kontrolü (Hungary)
        - Summer tatili: Haziran sonu - Ağustos sonu
        - Winter tatili: Aralık sonu - Ocak başı (yaklaşık 2 hafta)
        - Bahar tatili: Nisan ortası (yaklaşık 1 hafta)
        """
        month = date.month
        day = date.day
        
        # Summer tatili: 15 Haziran - 31 Ağustos
        if (month == 6 and day >= 15) or month in [7, 8] or (month == 9 and day <= 5):
            return True
        
        # Winter tatili: 20 Aralık - 5 Ocak
        if (month == 12 and day >= 20) or (month == 1 and day <= 5):
            return True
        
        # Bahar tatili: 10-20 Nisan arası
        if month == 4 and 10 <= day <= 20:
            return True
        
        return False
    
    def generate_calendar(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Detaylı calendar özellikleri generate
        
        Args:
            start_date: Start tarihi
            end_date: End tarihi
        
        Returns:
            DataFrame with detailed calendar features (UTC timezone, 15-minute resolution)
        """
        # Tarihleri UTC'ye çevir
        if start_date.tzinfo is None:
            start_date_utc = pd.Timestamp(start_date).tz_localize('UTC')
        else:
            start_date_utc = pd.Timestamp(start_date).tz_convert('UTC')
        
        if end_date.tzinfo is None:
            end_date_utc = pd.Timestamp(end_date).tz_localize('UTC')
        else:
            end_date_utc = pd.Timestamp(end_date).tz_convert('UTC')
        
        # Start: İlk günün başlangıcı (00:00)
        start_date_utc = start_date_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        # End: Son günün sonu (23:45)
        end_date_utc = end_date_utc.replace(hour=23, minute=45, second=0, microsecond=0)
        
        # 15 dakikalık zaman ızgarası generate (UTC)
        datetime_index = pd.date_range(
            start=start_date_utc,
            end=end_date_utc,
            freq='15min',
            tz='UTC'
        )
        
        logger.info(f"Takvim generateuluyor: {len(datetime_index):,} rows")
        
        # Calendar DataFrame generate
        calendar_df = pd.DataFrame({'datetime': datetime_index})
        
        # UTC'den Hungary local time'a çevir
        calendar_df['datetime_local'] = calendar_df['datetime'].dt.tz_convert(self.local_tz)
        
        # Temel tarih özellikleri
        calendar_df['date'] = calendar_df['datetime_local'].dt.date
        calendar_df['year'] = calendar_df['datetime_local'].dt.year
        calendar_df['month'] = calendar_df['datetime_local'].dt.month
        calendar_df['day'] = calendar_df['datetime_local'].dt.day
        calendar_df['day_of_week'] = calendar_df['datetime_local'].dt.dayofweek  # 0=Monday, 6=Sunday
        calendar_df['day_name'] = calendar_df['datetime_local'].dt.day_name()
        calendar_df['day_of_year'] = calendar_df['datetime_local'].dt.dayofyear
        
        # Hours özellikleri (local time)
        calendar_df['hour_local'] = calendar_df['datetime_local'].dt.hour
        calendar_df['minute'] = calendar_df['datetime_local'].dt.minute
        
        # ============================================
        # I. SIKLIK VE YAPISAL ÖZELLİKLER (Cyclical)
        # ============================================
        
        # hour_sin, hour_cos: Yerel saat (0-23) döngüsü
        calendar_df['hour_sin'] = np.sin(2 * np.pi * calendar_df['hour_local'] / 24)
        calendar_df['hour_cos'] = np.cos(2 * np.pi * calendar_df['hour_local'] / 24)
        
        # q_of_h_sin, q_of_h_cos: 15 dakikalık dilimin saat içindeki döngüsü (0, 15, 30, 45)
        minute_interval = calendar_df['minute'] // 15  # 0, 1, 2, 3
        calendar_df['q_of_h_sin'] = np.sin(2 * np.pi * minute_interval / 4)
        calendar_df['q_of_h_cos'] = np.cos(2 * np.pi * minute_interval / 4)
        
        # weekday_sin, weekday_cos: Haftalık ritmin sürekliliği (Monday, Pazar'a komşudur)
        calendar_df['weekday_sin'] = np.sin(2 * np.pi * calendar_df['day_of_week'] / 7)
        calendar_df['weekday_cos'] = np.cos(2 * np.pi * calendar_df['day_of_week'] / 7)
        
        # month_sin, month_cos: Aylık mevsimsel döngünün sürekliliği
        calendar_df['month_sin'] = np.sin(2 * np.pi * calendar_df['month'] / 12)
        calendar_df['month_cos'] = np.cos(2 * np.pi * calendar_df['month'] / 12)
        
        # day_sin, day_cos: Yıllık mevsimsel eğilimin sürekliliği (Yılın Günü)
        max_day_of_year = calendar_df['day_of_year'].max()
        calendar_df['day_sin'] = np.sin(2 * np.pi * calendar_df['day_of_year'] / max_day_of_year)
        calendar_df['day_cos'] = np.cos(2 * np.pi * calendar_df['day_of_year'] / max_day_of_year)
        
        # ============================================
        # II. DAVRANIŞSAL VE KATEGORİK ÖZELLİKLER
        # ============================================
        
        # Binary özellikler
        
        # is_dst: Summer Hoursi Uygulaması aktif mi? (Hours bazında, her rows için farklı olabilir)
        calendar_df['is_dst'] = calendar_df['datetime_local'].apply(
            lambda x: 1 if x.dst() != timedelta(0) else 0
        )
        
        # Gün bazında calculatenacak özellikler için yardımcı fonksiyonlar
        def is_weekend_or_holiday(date_obj):
            """Holiday veya hafta sonu kontrolü"""
            if date_obj in self.hungary_holidays:
                return 1
            weekday = pd.Timestamp(date_obj).dayofweek
            if weekday in [5, 6]:  # Cumartesi, Pazar
                return 1
            return 0
        
        # Gün bazında tüm özellikleri calculate (her gün için bir kez)
        unique_dates = calendar_df['date'].unique()
        day_features = {}
        
        for date_obj in unique_dates:
            weekday = pd.Timestamp(date_obj).dayofweek
            
            # is_holiday: Hungary resmi tatili mi?
            is_holiday = 1 if date_obj in self.hungary_holidays else 0
            
            # is_weekend: Cumartesi veya Pazar mı?
            is_weekend = 1 if weekday in [5, 6] else 0
            
            # is_workday: Holiday veya hafta sonu olmayan, normal çalışma günü mü?
            is_workday = 1 if (is_holiday == 0 and is_weekend == 0) else 0
            
            # is_school_holiday: Okul ve üniversite tatili dönemleri mi?
            is_school_holiday = 1 if self._is_school_holiday(date_obj) else 0
            
            # Önceki ve sonraki günler
            prev_date = date_obj - timedelta(days=1)
            next_date = date_obj + timedelta(days=1)
            
            # before_holiday_1day: Holidayden önceki son iş günü mü?
            next_is_holiday = 1 if next_date in self.hungary_holidays else 0
            before_holiday_1day = 1 if (next_is_holiday == 1 and is_workday == 1) else 0
            
            # after_holiday_1day: Holidayden sonraki ilk iş günü mü?
            prev_is_holiday = 1 if prev_date in self.hungary_holidays else 0
            after_holiday_1day = 1 if (prev_is_holiday == 1 and is_workday == 1) else 0
            
            # is_bridge_day: Normal tanım - Önceki gün tatil/hafta sonu VE sonraki gün tatil/hafta sonu, bugün iş günü
            prev_is_off = is_weekend_or_holiday(prev_date)
            next_is_off = is_weekend_or_holiday(next_date)
            is_bridge_day = 1 if (prev_is_off == 1 and next_is_off == 1 and is_workday == 1) else 0
            
            # is_payday: Ayın 1'i veya 15'i civarı mı? (1, 2, 3, 14, 15, 16)
            day = date_obj.day
            is_payday = 1 if day in [1, 2, 3, 14, 15, 16] else 0
            
            # christmas_period: 24–26 Aralık dönemi mi?
            month = date_obj.month
            christmas_period = 1 if (month == 12 and day in [24, 25, 26]) else 0
            
            # natl_day_peak_20Aug: 20 Ağustos (Aziz Stephen Günü) mü?
            natl_day_peak_20Aug = 1 if (month == 8 and day == 20) else 0
            
            # All özellikleri dictionary'ye save
            day_features[date_obj] = {
                'is_holiday': is_holiday,
                'is_weekend': is_weekend,
                'is_workday': is_workday,
                'is_school_holiday': is_school_holiday,
                'before_holiday_1day': before_holiday_1day,
                'after_holiday_1day': after_holiday_1day,
                'is_bridge_day': is_bridge_day,
                'is_payday': is_payday,
                'christmas_period': christmas_period,
                'natl_day_peak_20Aug': natl_day_peak_20Aug
            }
        
        # Gün bazında calculatenan özellikleri tüm rowslara uygula
        for feature_name in ['is_holiday', 'is_weekend', 'is_workday', 'is_school_holiday',
                            'before_holiday_1day', 'after_holiday_1day', 'is_bridge_day',
                            'is_payday', 'christmas_period', 'natl_day_peak_20Aug']:
            calendar_df[feature_name] = calendar_df['date'].apply(
                lambda x: day_features[x][feature_name]
            )
        
        # is_work_hour: Hafta içi, 08:00–18:00 arası ana mesai saatleri mi?
        calendar_df['is_work_hour'] = (
            (calendar_df['hour_local'] >= self.business_hours_start) & 
            (calendar_df['hour_local'] < self.business_hours_end) &
            (calendar_df['is_workday'] == 1)
        ).astype(int)
        
        # is_lunch_hour: Hafta içi, 12:00–14:00 arası öğle yemeği saati mi?
        calendar_df['is_lunch_hour'] = (
            (calendar_df['hour_local'] >= 12) & 
            (calendar_df['hour_local'] < 14) &
            (calendar_df['is_workday'] == 1)
        ).astype(int)
        
        # is_peak_hour: Hafta içi, 18:00–22:00 arası akşam piki saatleri mi?
        calendar_df['is_peak_hour'] = (
            (calendar_df['hour_local'] >= 18) & 
            (calendar_df['hour_local'] < 22) &
            (calendar_df['is_workday'] == 1)
        ).astype(int)
        
        # is_off_peak_hour: 00:00–06:00 arası en düşük taban yük saatleri mi?
        calendar_df['is_off_peak_hour'] = (
            (calendar_df['hour_local'] >= 0) & 
            (calendar_df['hour_local'] < 6)
        ).astype(int)
        
        # is_shoulder_hour: Kritik geçiş saatleri (Sabah 06:00–08:00 ve Gece 22:00–24:00 arası)
        calendar_df['is_shoulder_hour'] = (
            ((calendar_df['hour_local'] >= 6) & (calendar_df['hour_local'] < 8)) |
            ((calendar_df['hour_local'] >= 22) & (calendar_df['hour_local'] < 24))
        ).astype(int)
        
        # ============================================
        # III. BEKLENTİ VE ÖZEL GÜN ÖZELLİKLERİ
        # (Yukarıda gün bazında calculatendı, burada sadece saat bazlı özellikler var)
        # ============================================
        
        # ============================================
        # ONE-HOT ENCODING ÖZELLİKLERİ
        # ============================================
        
        # day_type_OHE: Günün tipi (Normal Weekday, Monday, Weekend, Holiday)
        def get_day_type(row):
            if row['is_holiday'] == 1:
                return 'Holiday'
            elif row['is_weekend'] == 1:
                return 'Weekend'
            elif row['day_of_week'] == 0:  # Monday
                return 'Monday'
            else:
                return 'Normal_Weekday'
        
        calendar_df['day_type'] = calendar_df.apply(get_day_type, axis=1)
        
        # One-Hot Encoding için dummy variables
        day_type_dummies = pd.get_dummies(calendar_df['day_type'], prefix='day_type_OHE')
        calendar_df = pd.concat([calendar_df, day_type_dummies], axis=1)
        calendar_df.drop('day_type', axis=1, inplace=True)
        
        # season_flag_OHE: Season (Winter, Spring, Summer, Autumn)
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        calendar_df['season'] = calendar_df['month'].apply(get_season)
        season_dummies = pd.get_dummies(calendar_df['season'], prefix='season_flag_OHE')
        calendar_df = pd.concat([calendar_df, season_dummies], axis=1)
        calendar_df.drop('season', axis=1, inplace=True)
        
        # Sütun sırasını düzenle (datetime ilk sütun)
        # Önce siklik özellikler, sonra binary, sonra OHE
        priority_cols = ['datetime']
        
        # Siklik özellikler
        cyclic_cols = ['hour_sin', 'hour_cos', 'q_of_h_sin', 'q_of_h_cos', 
                      'weekday_sin', 'weekday_cos', 'month_sin', 'month_cos', 
                      'day_sin', 'day_cos']
        
        # Binary özellikler
        binary_cols = ['is_dst', 'is_holiday', 'is_weekend', 'is_workday', 'is_work_hour',
                      'is_lunch_hour', 'is_peak_hour', 'is_off_peak_hour', 'is_shoulder_hour',
                      'is_school_holiday', 'before_holiday_1day', 'after_holiday_1day',
                      'is_bridge_day', 'is_payday', 'christmas_period', 'natl_day_peak_20Aug']
        
        # OHE özellikler
        ohe_cols = [col for col in calendar_df.columns if 'OHE' in col]
        
        # Diğer özellikler
        other_cols = [col for col in calendar_df.columns 
                     if col not in priority_cols + cyclic_cols + binary_cols + ohe_cols]
        
        # Sütun sırası
        ordered_cols = priority_cols + cyclic_cols + binary_cols + ohe_cols + other_cols
        calendar_df = calendar_df[[col for col in ordered_cols if col in calendar_df.columns]]
        
        logger.info(f"✅ Takvim generateuldu: {len(calendar_df):,} rows, {len(calendar_df.columns)} özellik")
        logger.info(f"   - Siklik özellikler: {len(cyclic_cols)}")
        logger.info(f"   - Binary özellikler: {len(binary_cols)}")
        logger.info(f"   - OHE özellikler: {len(ohe_cols)}")
        
        return calendar_df
    
    def save_csv(self, df: pd.DataFrame, filepath: str):
        """CSV'ye save"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"✅ Takvim saved: {filepath}")


if __name__ == "__main__":
    generator = HungaryCalendarGenerator()
    
    start = datetime(2015, 1, 1)
    end = datetime(2024, 12, 31)
    
    # Takvim generate
    calendar_df = generator.generate_calendar(start, end)
    
    # Kaydet
    generator.save_csv(calendar_df, "data/raw/hungary_calendar_2015_2024.csv")
