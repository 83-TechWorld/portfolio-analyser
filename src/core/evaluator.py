import numpy as np

class PerformanceEvaluator:
    """Performance evaluation and metrics calculation"""
    
    def calculate_portfolio_pnl(self, X, Y, M):
        """Calculate daily P&L: (X @ M ⊙ Y) @ 1"""
        try:
            W = X @ M  # Portfolio weights over time
            pnl_daily = np.sum(W * Y, axis=1)  # Element-wise product then sum
            return pnl_daily
        except Exception as e:
            print(f"P&L calculation error: {e}")
            return np.array([])
    
    def sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns) * 252  # Annualized return
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        return (mean_return - risk_free_rate) / volatility
    
    def comprehensive_metrics(self, returns):
        """Calculate comprehensive performance metrics"""
        if len(returns) == 0:
            return {}
        
        metrics = {
            'total_return': np.sum(returns),
            'annualized_return': np.mean(returns) * 252,
            'annualized_volatility': np.std(returns) * np.sqrt(252),
            'sharpe_ratio': self.sharpe_ratio(returns),
            'max_drawdown': self.max_drawdown(returns),
            'win_rate': np.mean(returns > 0),
            'best_day': np.max(returns),
            'worst_day': np.min(returns)
        }
        return metrics
    
    def max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak)
        return np.min(drawdown)
    
    def evaluate_strategy(self, X_train, Y_train, X_test, Y_test, M, strategy_name):
        """Complete strategy evaluation"""
        results = {'strategy': strategy_name}
        
        # In-sample evaluation
        pnl_train = self.calculate_portfolio_pnl(X_train, Y_train, M)
        train_metrics = self.comprehensive_metrics(pnl_train)
        results['in_sample'] = train_metrics
        
        # Out-of-sample evaluation
        pnl_test = self.calculate_portfolio_pnl(X_test, Y_test, M)
        test_metrics = self.comprehensive_metrics(pnl_test)
        results['out_sample'] = test_metrics
        
        return results