#!/usr/bin/env python3
"""
TNO FCI Raw Data Collector
Manages TradingView raw data files and prompts for monthly updates
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import argparse

class RawDataCollector:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
        # Define all TradingView data sources with their paths
        self.data_sources = {
            'ISM PMI': {
                'raw_path': 'raw_all',
                'file_pattern': 'ECONOMICS_USBCOI',
                'symbol': 'ECONOMICS:USBCOI',
                'timeframe': '1M'
            },
            'Fed Rates': {
                'raw_path': 'raw_all',
                'file_pattern': 'ECONOMICS_USINTR',
                'symbol': 'ECONOMICS:USINTR',
                'timeframe': '1M'
            },
            'Dollar Index': {
                'raw_path': 'raw_all',
                'file_pattern': 'TVC_DXY',
                'symbol': 'TVC:DXY',
                'timeframe': '1M'
            },
            'Gold': {
                'raw_path': 'raw_all',
                'file_pattern': 'TVC_GOLD',
                'symbol': 'TVC:GOLD',
                'timeframe': '1M'
            },
            'Oil': {
                'raw_path': 'raw_all',
                'file_pattern': 'TVC_USOIL',
                'symbol': 'TVC:USOIL',
                'timeframe': '1M'
            },
            'Silver': {
                'raw_path': 'raw_all',
                'file_pattern': 'TVC_SILVER',
                'symbol': 'TVC:SILVER',
                'timeframe': '1M'
            }
        }
        
    def find_raw_file(self, source_info):
        """Find the raw data file for a given source"""
        raw_dir = os.path.join(self.data_dir, source_info['raw_path'])
        if not os.path.exists(raw_dir):
            return None
            
        # Look for files matching the pattern
        for filename in os.listdir(raw_dir):
            if filename.startswith(source_info['file_pattern']) and filename.endswith('.csv'):
                return os.path.join(raw_dir, filename)
        return None
        
    def check_last_date(self, filepath):
        """Check the last date in a CSV file"""
        try:
            # Read the CSV - TradingView format has 'time' as first column
            df = pd.read_csv(filepath)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                last_date = df['time'].max()
                return last_date
            else:
                # Try first column as date
                df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                return df.iloc[:, 0].max()
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            return None
            
    def tradingview_api_download(self, symbol, timeframe, start_date, end_date):
        """
        STUB: TradingView API download function
        
        In the future, this would connect to TradingView API and download data
        For now, it returns None to trigger manual download
        
        Parameters:
        - symbol: TradingView symbol (e.g., 'TVC:GOLD')
        - timeframe: Timeframe (e.g., '1M' for monthly)
        - start_date: Start date for data download
        - end_date: End date for data download
        
        Returns:
        - DataFrame with new data or None if not implemented
        """
        # TODO: Implement TradingView API connection here
        # For now, return None to trigger manual download
        return None
        
    def append_new_data(self, filepath, new_data):
        """Append new data to existing file"""
        try:
            # Read existing data
            existing = pd.read_csv(filepath)
            
            # Append new data
            combined = pd.concat([existing, new_data], ignore_index=True)
            
            # Remove duplicates based on time column
            if 'time' in combined.columns:
                combined['time'] = pd.to_datetime(combined['time'])
                combined = combined.drop_duplicates(subset=['time'], keep='last')
                combined = combined.sort_values('time')
            
            # Save back
            combined.to_csv(filepath, index=False)
            return len(new_data)
        except Exception as e:
            print(f"  ✗ Error appending data: {e}")
            return 0
            
    def check_and_update_source(self, name, source_info):
        """Check and update a single data source"""
        print(f"\n{'='*60}")
        print(f"Checking {name}")
        print(f"{'='*60}")
        
        # Find the raw file
        raw_file = self.find_raw_file(source_info)
        
        if not raw_file:
            print(f"✗ No raw data file found!")
            print(f"  Expected in: {os.path.join(self.data_dir, source_info['raw_path'])}")
            print(f"  Pattern: {source_info['file_pattern']}*.csv")
            print(f"\n📥 MANUAL ACTION REQUIRED:")
            print(f"  1. Go to TradingView.com")
            print(f"  2. Search for symbol: {source_info['symbol']}")
            print(f"  3. Set timeframe to: {source_info['timeframe']} (Monthly)")
            print(f"  4. Export data as CSV")
            print(f"  5. Save to: {os.path.join(self.data_dir, source_info['raw_path'])}")
            return
            
        print(f"✓ Found raw file: {os.path.basename(raw_file)}")
        
        # Check last date
        last_date = self.check_last_date(raw_file)
        if not last_date:
            print(f"✗ Could not read last date from file")
            return
            
        print(f"  Last data point: {last_date.strftime('%Y-%m-%d')}")
        
        # Calculate days since last update
        days_old = (datetime.now() - last_date).days
        
        # Check if update is needed (data older than 30 days)
        if days_old < 30:
            print(f"✓ Data is up to date ({days_old} days old)")
            return
            
        print(f"⚠ Data is {days_old} days old - update needed")
        
        # Try to download via API (stub for now)
        start_date = last_date + timedelta(days=1)
        end_date = datetime.now()
        
        print(f"\nAttempting to download data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        new_data = self.tradingview_api_download(
            source_info['symbol'],
            source_info['timeframe'],
            start_date,
            end_date
        )
        
        if new_data is not None and not new_data.empty:
            # Append new data
            records_added = self.append_new_data(raw_file, new_data)
            print(f"✓ Added {records_added} new records")
        else:
            # Manual download required
            print(f"\n📥 MANUAL ACTION REQUIRED:")
            print(f"  1. Go to TradingView.com")
            print(f"  2. Search for symbol: {source_info['symbol']}")
            print(f"  3. Set timeframe to: {source_info['timeframe']} (Monthly)")
            print(f"  4. Export ONLY new data from {start_date.strftime('%Y-%m-%d')} onwards")
            print(f"  5. Either:")
            print(f"     a) Replace the existing file in: {os.path.dirname(raw_file)}")
            print(f"     b) Manually append the new data to: {os.path.basename(raw_file)}")
            
    def update_all(self):
        """Check and update all data sources"""
        print("\n" + "="*70)
        print("TNO FCI RAW DATA COLLECTOR")
        print("="*70)
        print("\nThis script checks TradingView raw data files and prompts for updates.")
        print("Run this once a month to keep your data current.")
        
        # Check each data source
        for name, source_info in self.data_sources.items():
            self.check_and_update_source(name, source_info)
            
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\n✓ Raw data check complete!")
        print("\nNext steps:")
        print("1. If any manual downloads were required, complete them")
        print("2. Run '2_prep_raw_data.py' to process the raw data")
        print("3. Run '3_tno_fci_calculator.py' to calculate the FCI")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check and update TNO FCI raw data from TradingView')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    args = parser.parse_args()
    
    collector = RawDataCollector(data_dir=args.data_dir)
    collector.update_all()