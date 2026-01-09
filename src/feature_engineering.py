"""Feature Engineering - Lag & Rolling Window Features

This script creates lag and rolling statistics (mean/std/min/max) from the load time series.
Important detail: we avoid data leakage by only using past values (e.g., shift before rolling).
It also merges the prepared load, weather, and calendar datasets into one final table."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Lag ve rolling window özellikleri generateucu"""
    
    def __init__(self):
        self.lag_hours = [48, 72, 96, 120, 144, 168]  # 2-7 gün
        self.window_hours = [48, 72, 96, 120, 144, 168]  # 2-7 gün
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'load_MW') -> pd.DataFrame:
        """
        Lag özellikleri generate (geçmiş değerler)
        
        Args:
            df: DataFrame (datetime index olmalı, UTC timezone)
            target_col: Lag generateulacak sütun adı
        
        Returns:
            Lag özellikleri eklenmiş DataFrame
        """
        logger.info(f"Lag özellikleri generateuluyor: {self.lag_hours} saat")
        
        # DataFrame'i kopyala
        df_lagged = df.copy()
        
        # Datetime'ı index yap (eğer değilse)
        if 'datetime' in df_lagged.columns:
            df_lagged = df_lagged.set_index('datetime')
        
        # Datetime index UTC timezone'da olmalı
        if df_lagged.index.tz is None:
            df_lagged.index = pd.to_datetime(df_lagged.index).tz_localize('UTC')
        elif df_lagged.index.tz != pd.Timedelta(hours=0):
            df_lagged.index = df_lagged.index.tz_convert('UTC')
        
        # Sıralı olduğundan emin ol
        df_lagged = df_lagged.sort_index()
        
        # Her lag için özellik generate
        for lag_hours in self.lag_hours:
            # 15 dakikalık aralıklar için lag sayısı
            lag_periods = lag_hours * 4  # 1 saat = 4 * 15 dakika
            
            lag_col_name = f'{target_col}_lag_{lag_hours}h'
            df_lagged[lag_col_name] = df_lagged[target_col].shift(lag_periods)
            
            logger.info(f"  ✓ {lag_col_name} generateuldu")
        
        # Index'i tekrar sütun yap
        df_lagged = df_lagged.reset_index()
        
        return df_lagged
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'load_MW') -> pd.DataFrame:
        """
        Rolling window özellikleri generate (mean, std, min, max)
        Leakage olmadan: sadece geçmiş dataler kullanılır
        
        Args:
            df: DataFrame (datetime index olmalı, UTC timezone)
            target_col: Rolling window generateulacak sütun adı
        
        Returns:
            Rolling window özellikleri eklenmiş DataFrame
        """
        logger.info(f"Rolling window özellikleri generateuluyor: {self.window_hours} saat")
        
        # DataFrame'i kopyala
        df_rolling = df.copy()
        
        # Datetime'ı index yap (eğer değilse)
        if 'datetime' in df_rolling.columns:
            df_rolling = df_rolling.set_index('datetime')
        
        # Datetime index UTC timezone'da olmalı
        if df_rolling.index.tz is None:
            df_rolling.index = pd.to_datetime(df_rolling.index).tz_localize('UTC')
        elif df_rolling.index.tz != pd.Timedelta(hours=0):
            df_rolling.index = df_rolling.index.tz_convert('UTC')
        
        # Sıralı olduğundan emin ol
        df_rolling = df_rolling.sort_index()
        
        # Her window için özellik generate
        for window_hours in self.window_hours:
            # 15 dakikalık aralıklar için window size
            window_size = window_hours * 4  # 1 saat = 4 * 15 dakika
            
            # Rolling window (sadece geçmiş dataler - shift(1) ile leakage önlenir)
            # shift(1): O anki datayi dahil etmez, sadece geçmiş dataleri kullanır
            rolling = df_rolling[target_col].shift(1).rolling(
                window=window_size,
                min_periods=window_size  # En az window_size kadar data olsun (tam window için)
            )
            
            # Mean
            df_rolling[f'{target_col}_rolling_mean_{window_hours}h'] = rolling.mean()
            
            # Std
            df_rolling[f'{target_col}_rolling_std_{window_hours}h'] = rolling.std()
            
            # Min
            df_rolling[f'{target_col}_rolling_min_{window_hours}h'] = rolling.min()
            
            # Max
            df_rolling[f'{target_col}_rolling_max_{window_hours}h'] = rolling.max()
            
            logger.info(f"  ✓ {window_hours}h rolling window özellikleri generateuldu (mean, std, min, max)")
        
        # Index'i tekrar sütun yap
        df_rolling = df_rolling.reset_index()
        
        return df_rolling
    
    def merge_all_datasets(self, 
                          load_path: str,
                          weather_path: str,
                          calendar_path: str,
                          output_path: str = None) -> pd.DataFrame:
        """
        All datasetleri merge (load, weather, calendar)
        
        Args:
            load_path: Load datasi CSV dosya yolu
            weather_path: Weather datasi CSV dosya yolu
            calendar_path: Calendar datasi CSV dosya yolu
            output_path: Birleştirilmiş datayi savemek için dosya yolu (opsiyonel)
        
        Returns:
            Birleştirilmiş DataFrame
        """
        logger.info("Datasetler mergeiliyor...")
        
        # Load datasini oku
        logger.info(f"Load datasi okunuyor: {load_path}")
        df_load = pd.read_csv(load_path)
        df_load['datetime'] = pd.to_datetime(df_load['datetime'], utc=True)
        logger.info(f"  ✓ {len(df_load):,} rows")
        
        # Weather datasini oku
        logger.info(f"Weather datasi okunuyor: {weather_path}")
        df_weather = pd.read_csv(weather_path)
        df_weather['datetime'] = pd.to_datetime(df_weather['datetime'], utc=True)
        logger.info(f"  ✓ {len(df_weather):,} rows")
        
        # Calendar datasini oku
        logger.info(f"Calendar datasi okunuyor: {calendar_path}")
        df_calendar = pd.read_csv(calendar_path)
        df_calendar['datetime'] = pd.to_datetime(df_calendar['datetime'], utc=True)
        logger.info(f"  ✓ {len(df_calendar):,} rows")
        
        # Load ve Weather merge
        logger.info("Load ve Weather mergeiliyor...")
        df_merged = pd.merge(df_load, df_weather, on='datetime', how='inner')
        logger.info(f"  ✓ {len(df_merged):,} rows")
        
        # Calendar ile merge
        logger.info("Calendar ile mergeiliyor...")
        df_merged = pd.merge(df_merged, df_calendar, on='datetime', how='inner')
        logger.info(f"  ✓ {len(df_merged):,} rows")
        
        # Datetime'a göre sırala
        df_merged = df_merged.sort_values('datetime').reset_index(drop=True)
        
        logger.info(f"✅ All datasetler mergeildi: {len(df_merged):,} rows, {len(df_merged.columns)} sütun")
        
        # Kaydet (eğer path dataldiyse)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df_merged.to_csv(output_path, index=False)
            logger.info(f"✅ Birleştirilmiş data saved: {output_path}")
        
        return df_merged


if __name__ == "__main__":
    engineer = FeatureEngineer()
    
    # Load datasini oku
    load_path = "data/raw/hungary_load_data_2015_2024.csv"
    df_load = pd.read_csv(load_path)
    df_load['datetime'] = pd.to_datetime(df_load['datetime'], utc=True)
    
    # Lag özellikleri generate
    df_load = engineer.create_lag_features(df_load, target_col='load_MW')
    
    # Rolling window özellikleri generate
    df_load = engineer.create_rolling_features(df_load, target_col='load_MW')
    
    # Load datasini save (lag ve rolling özellikleri ile)
    output_load_path = "data/processed/hungary_load_with_features_2015_2024.csv"
    Path(output_load_path).parent.mkdir(parents=True, exist_ok=True)
    df_load.to_csv(output_load_path, index=False)
    logger.info(f"✅ Load datasi saved: {output_load_path}")
    
    # All datasetleri merge
    merged_df = engineer.merge_all_datasets(
        load_path=output_load_path,
        weather_path="data/raw/hungary_weather_2015_2024.csv",
        calendar_path="data/raw/hungary_calendar_2015_2024.csv",
        output_path="data/processed/hungary_merged_dataset_2015_2024.csv"
    )

