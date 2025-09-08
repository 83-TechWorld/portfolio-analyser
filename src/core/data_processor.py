import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PortfolioDataProcessor:
    """Core data processing without LLM dependencies"""
    
    def __init__(self):
        self.scaler_signals = StandardScaler()
        self.scaler_returns = StandardScaler()
        self.pca_signals = PCA(n_components=50)
        self.pca_returns = PCA(n_components=20)
        self.data_loaded = False
        self.raw_data = None
        self.processed_data = None
    
    def load_excel_data(self, filepath):
        """Load all sheets from Excel file"""
        try:
            sheet_names = ['In-sample returns', 'In-sample signals', 
                          'Out-sample returns', 'Out-sample signals']
            
            data = {}
            for sheet in sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet)
                key = sheet.lower().replace('-', '_').replace(' ', '_')
                data[key] = df
            
            self.raw_data = data
            self.data_loaded = True
            return True, f"Loaded {len(data)} sheets successfully"
        
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def preprocess_data(self):
        """Process raw data into ML-ready format"""
        if not self.data_loaded:
            return False, "No data loaded"
        
        try:
            processed = {}
            
            # Extract matrices (remove Day column)
            processed['X_train'] = self.raw_data['in_sample_signals'].iloc[:, 1:].values
            processed['Y_train'] = self.raw_data['in_sample_returns'].iloc[:, 1:].values
            processed['X_test'] = self.raw_data['out_sample_signals'].iloc[:, 1:].values
            processed['Y_test'] = self.raw_data['out_sample_returns'].iloc[:, 1:].values
            
            # Get asset and signal names
            processed['asset_names'] = list(self.raw_data['in_sample_returns'].columns[1:])
            processed['signal_names'] = list(self.raw_data['in_sample_signals'].columns[1:])
            
            # Handle missing values
            for key in ['X_train', 'Y_train', 'X_test', 'Y_test']:
                processed[key] = np.nan_to_num(processed[key], nan=0.0)
            
            # Standardize signals
            processed['X_train_scaled'] = self.scaler_signals.fit_transform(processed['X_train'])
            processed['X_test_scaled'] = self.scaler_signals.transform(processed['X_test'])
            
            self.processed_data = processed
            return True, f"Processed data: {processed['X_train'].shape}"
        
        except Exception as e:
            return False, f"Error processing data: {str(e)}"
    
    def get_data_summary(self):
        """Return data summary statistics"""
        if not self.processed_data:
            return "No processed data available"
        
        data = self.processed_data
        summary = {
            'training_days': data['X_train'].shape[0],
            'num_assets': data['X_train'].shape[1] if 'X_train' in data else data['Y_train'].shape[1],
            'num_signals': data['X_train'].shape[1],
            'test_days': data['X_test'].shape[0],
            'asset_names': data['asset_names'][:5],  # First 5 assets
            'signal_names': data['signal_names'][:5]  # First 5 signals
        }
        return summary