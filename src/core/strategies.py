from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import numpy as np
from sklearn.decomposition import PCA

class StrategyEngine:
    """All portfolio strategies implementation"""
    
    def __init__(self):
        self.strategies = {}
        self.strategy_matrices = {}
    
    def benchmark_ols(self, X, Y):
        """Benchmark OLS Strategy: M = (X'X)^(-1)X'Y"""
        try:
            # Use pseudo-inverse for numerical stability
            XtX_inv = np.linalg.pinv(X.T @ X + 1e-8 * np.eye(X.shape[1]))
            M = XtX_inv @ X.T @ Y
            self.strategy_matrices['benchmark'] = M
            return M, "OLS strategy computed successfully"
        except Exception as e:
            return None, f"OLS error: {str(e)}"
    
    def revised_cca_strategy(self, X, Y, n_components=10):
        """CCA-based Revised Strategy"""
        try:
            # Step 1: PCA preprocessing
            pca_x = PCA(n_components=min(50, X.shape[1]))
            pca_y = PCA(n_components=min(20, Y.shape[1]))
            
            X_pca = pca_x.fit_transform(X)
            Y_pca = pca_y.fit_transform(Y)
            
            # Step 2: Canonical Correlation Analysis
            cca = CCA(n_components=n_components)
            X_c, Y_c = cca.fit_transform(X_pca, Y_pca)
            
            # Step 3: SVD for low-rank approximation
            U, s, Vt = np.linalg.svd(X_c.T @ Y_c, full_matrices=False)
            
            # Step 4: Construct transformation matrix
            # Transform back to original space
            M = pca_x.components_.T @ U @ Vt @ pca_y.components_
            
            self.strategy_matrices['revised_cca'] = M
            return M, "CCA strategy computed successfully"
        
        except Exception as e:
            return None, f"CCA error: {str(e)}"
    
    def ridge_strategy(self, X, Y, alpha=0.01):
        """Ridge Regression Strategy"""
        try:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X, Y)
            M = ridge.coef_.T
            self.strategy_matrices['ridge'] = M
            return M, f"Ridge strategy (α={alpha}) computed successfully"
        except Exception as e:
            return None, f"Ridge error: {str(e)}"
    
    def lasso_strategy(self, X, Y, alpha=0.01):
        """Lasso Regression Strategy"""
        try:
            lasso = Lasso(alpha=alpha, max_iter=2000)
            lasso.fit(X, Y)
            M = lasso.coef_.T
            self.strategy_matrices['lasso'] = M
            return M, f"Lasso strategy (α={alpha}) computed successfully"
        except Exception as e:
            return None, f"Lasso error: {str(e)}"
    
    def elastic_net_strategy(self, X, Y, alpha=0.01, l1_ratio=0.5):
        """Elastic Net Strategy"""
        try:
            elastic = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
            elastic.fit(X, Y)
            M = elastic.coef_.T
            self.strategy_matrices['elastic_net'] = M
            return M, f"Elastic Net (α={alpha}, l1={l1_ratio}) computed successfully"
        except Exception as e:
            return None, f"Elastic Net error: {str(e)}"