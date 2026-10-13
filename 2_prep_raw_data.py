#!/usr/bin/env python3
"""
TNO FCI Data Prep Script
Processes TradingView raw data files into clean, standardized format
"""

import pandas as pd
import os
import argparse

class DataPreprocessor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
        # Define data sources with their raw and clean paths
        self.data_sources = {
            'ISM PMI': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/ismpmi_clean.csv',
                'file_pattern': 'ECONOMICS_USBCOI',
                'column_name': 'ISM_PMI'
            },
            'Fed Rates': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/fedrates_clean.csv',
                'file_pattern': 'ECONOMICS_USINTR',
                'column_name': 'Fed_Rate'
            },
            'Dollar Index': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/dxy_clean.csv',
                'file_pattern': 'TVC_DXY',
                'column_name': 'Dollar_Index'
            },
            'Gold': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/gold_clean.csv',
                'file_pattern': 'TVC_GOLD',
                'column_name': 'Gold_Price'
            },
            'Oil': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/oil_clean.csv',
                'file_pattern': 'TVC_USOIL',
                'column_name': 'Oil_Price'
            },
            'Silver': {
                'raw_path': 'raw_all',
                'clean_path': 'clean_all/silver_clean.csv',
                'file_pattern': 'TVC_SILVER',
                'column_name': 'Silver_Price'
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
        
    def process_tradingview_file(self, filepath, column_name):
        """Process a TradingView CSV file into clean format"""
        try:
            # Read CSV
            df = pd.read_csv(filepath)
            
            print(f"  Original columns: {list(df.columns)}")
            print(f"  First few rows:")
            print(df.head(3))
            
            # Create clean dataframe
            clean_df = pd.DataFrame()
            
            # TradingView format typically has 'time' as first column
            if 'time' in df.columns:
                clean_df['Date'] = pd.to_datetime(df['time'])
                
                # For OHLC data, use close price; otherwise use the value column
                if 'close' in df.columns:
                    clean_df[column_name] = df['close']
                elif 'value' in df.columns:
                    clean_df[column_name] = df['value']
                else:
                    # Use the second column (first is time)
                    clean_df[column_name] = df.iloc[:, 1]
            else:
                # Fallback: assume first column is date
                clean_df['Date'] = pd.to_datetime(df.iloc[:, 0])
                
                # For OHLC data, use close price
                if 'close' in df.columns:
                    clean_df[column_name] = df['close']
                else:
                    # Use the second column
                    clean_df[column_name] = df.iloc[:, 1]
            
            # Set index and sort
            clean_df.set_index('Date', inplace=True)
            clean_df.sort_index(inplace=True)
            
            # Remove any NaN values
            clean_df = clean_df.dropna()
            
            return clean_df
            
        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
            return None
            
    def process_source(self, name, source_info):
        """Process a single data source"""
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")
        
        # Find raw file
        raw_file = self.find_raw_file(source_info)
        
        if not raw_file:
            print(f"✗ No raw data file found!")
            print(f"  Expected in: {os.path.join(self.data_dir, source_info['raw_path'])}")
            print(f"  Pattern: {source_info['file_pattern']}*.csv")
            print(f"  Run '1_data_collector.py' to check for raw data files")
            return False
            
        print(f"✓ Found raw file: {os.path.basename(raw_file)}")
        
        # Process the file
        clean_df = self.process_tradingview_file(raw_file, source_info['column_name'])
        
        if clean_df is None or clean_df.empty:
            print(f"✗ Failed to process raw data")
            return False
            
        # Save clean data
        clean_path = os.path.join(self.data_dir, source_info['clean_path'])
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        clean_df.to_csv(clean_path)
        
        print(f"✓ Saved clean data to: {source_info['clean_path']}")
        print(f"  Records: {len(clean_df)}")
        print(f"  Date range: {clean_df.index.min()} to {clean_df.index.max()}")
        
        return True
        
    def process_all(self):
        """Process all raw data files"""
        print("\n" + "="*70)
        print("TNO FCI DATA PREPROCESSOR")
        print("="*70)
        print("\nThis script processes TradingView raw data into clean format.")
        
        success_count = 0
        
        # Process each data source
        for name, source_info in self.data_sources.items():
            if self.process_source(name, source_info):
                success_count += 1
                
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n✓ Successfully processed {success_count}/{len(self.data_sources)} data sources")
        
        if success_count == len(self.data_sources):
            print("\nAll data processed successfully!")
            print("\nNext step: Run '3_tno_fci_calculator.py' to calculate the FCI")
        else:
            print("\n⚠ Some data sources failed to process")
            print("Check the errors above and ensure raw data files exist")
            
    def show_clean_data_summary(self):
        """Show summary of all clean data files"""
        print("\n" + "="*70)
        print("CLEAN DATA SUMMARY")
        print("="*70)
        
        for name, source_info in self.data_sources.items():
            clean_path = os.path.join(self.data_dir, source_info['clean_path'])
            
            if os.path.exists(clean_path):
                try:
                    df = pd.read_csv(clean_path, index_col=0, parse_dates=True)
                    if len(df) > 0:
                        print(f"\n{name}:")
                        print(f"  File: {source_info['clean_path']}")
                        print(f"  Records: {len(df)}")
                        print(f"  Date range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
                        print(f"  Latest value: {df.iloc[-1, 0]:.2f}")
                    else:
                        print(f"\n{name}: No data in file")
                except Exception as e:
                    print(f"\n{name}: Error reading file - {e}")
            else:
                print(f"\n{name}: File not found - {source_info['clean_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process TNO FCI raw data into clean format')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--summary', action='store_true', help='Show summary of clean data files')
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(data_dir=args.data_dir)
    
    if args.summary:
        preprocessor.show_clean_data_summary()
    else:
        preprocessor.process_all()
        preprocessor.show_clean_data_summary()