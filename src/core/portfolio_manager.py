from .data_processor import PortfolioDataProcessor
from .strategies import StrategyEngine
from .evaluator import PerformanceEvaluator

class PortfolioManager:
    """Main portfolio management system"""
    
    def __init__(self):
        self.data_processor = PortfolioDataProcessor()
        self.strategy_engine = StrategyEngine()
        self.evaluator = PerformanceEvaluator()
        self.results = {}
        self.current_data = None
    
    def load_data(self, filepath):
        """Load and process portfolio data"""
        success, message = self.data_processor.load_excel_data(filepath)
        if success:
            success, message = self.data_processor.preprocess_data()
            if success:
                self.current_data = self.data_processor.processed_data
        return success, message
    
    def run_strategy(self, strategy_name, **params):
        """Execute a specific strategy"""
        if not self.current_data:
            return False, "No data loaded"
        
        X_train = self.current_data['X_train_scaled']
        Y_train = self.current_data['Y_train']
        X_test = self.current_data['X_test_scaled']
        Y_test = self.current_data['Y_test']
        
        # Strategy dispatch
        if strategy_name == 'benchmark':
            M, message = self.strategy_engine.benchmark_ols(X_train, Y_train)
        elif strategy_name == 'revised_cca':
            M, message = self.strategy_engine.revised_cca_strategy(X_train, Y_train, **params)
        elif strategy_name == 'ridge':
            M, message = self.strategy_engine.ridge_strategy(X_train, Y_train, **params)
        elif strategy_name == 'lasso':
            M, message = self.strategy_engine.lasso_strategy(X_train, Y_train, **params)
        elif strategy_name == 'elastic_net':
            M, message = self.strategy_engine.elastic_net_strategy(X_train, Y_train, **params)
        else:
            return False, f"Unknown strategy: {strategy_name}"
        
        if M is not None:
            # Evaluate strategy performance
            results = self.evaluator.evaluate_strategy(X_train, Y_train, X_test, Y_test, M, strategy_name)
            self.results[strategy_name] = results
            return True, f"{message}\nOut-sample Sharpe: {results['out_sample']['sharpe_ratio']:.3f}"
        
        return False, message
    
    def get_asset_recommendation(self, current_signals=None, top_n=10):
        """Recommend assets based on current signals"""
        if not self.current_data or 'revised_cca' not in self.results:
            return "No strategy results available for recommendations"
        
        try:
            # Use latest signals if not provided
            if current_signals is None:
                current_signals = self.current_data['X_test_scaled'][-1]  # Latest day signals
            
            # Get strategy matrix
            M = self.strategy_engine.strategy_matrices['revised_cca']
            
            # Calculate expected returns: signal × strategy matrix
            expected_returns = current_signals @ M
            
            # Get asset names
            asset_names = self.current_data['asset_names']
            
            # Create recommendations
            recommendations = []
            for i, (asset, expected_return) in enumerate(zip(asset_names, expected_returns)):
                recommendations.append({
                    'asset': asset,
                    'expected_return': expected_return,
                    'rank': i + 1
                })
            
            # Sort by expected return
            recommendations.sort(key=lambda x: x['expected_return'], reverse=True)
            
            return recommendations[:top_n]
        
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def get_portfolio_allocation(self, risk_tolerance='medium'):
        """Generate optimal portfolio allocation"""
        recommendations = self.get_asset_recommendation(top_n=20)
        
        if isinstance(recommendations, str):  # Error message
            return recommendations
        
        # Risk-based allocation
        risk_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
        multiplier = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Simple allocation based on expected returns
        total_positive_returns = sum(max(0, rec['expected_return']) for rec in recommendations)
        
        if total_positive_returns > 0:
            allocations = []
            for rec in recommendations:
                if rec['expected_return'] > 0:
                    weight = (rec['expected_return'] / total_positive_returns) * multiplier
                    weight = min(weight, 0.15)  # Max 15% per asset
                    allocations.append({
                        'asset': rec['asset'],
                        'weight': weight,
                        'expected_return': rec['expected_return']
                    })
            
            # Normalize weights
            total_weight = sum(alloc['weight'] for alloc in allocations)
            for alloc in allocations:
                alloc['weight'] = alloc['weight'] / total_weight
            
            return allocations
        
        return "No positive expected returns found"
    
    def compare_strategies(self):
        """Compare all executed strategies"""
        if not self.results:
            return "No strategies executed yet"
        
        comparison = []
        for strategy_name, results in self.results.items():
            comparison.append({
                'Strategy': strategy_name.replace('_', ' ').title(),
                'In-Sample Sharpe': results['in_sample']['sharpe_ratio'],
                'Out-Sample Sharpe': results['out_sample']['sharpe_ratio'],
                'Out-Sample Return': results['out_sample']['annualized_return'],
                'Out-Sample Volatility': results['out_sample']['annualized_volatility'],
                'Max Drawdown': results['out_sample']['max_drawdown']
            })
        
        return sorted(comparison, key=lambda x: x['Out-Sample Sharpe'], reverse=True)