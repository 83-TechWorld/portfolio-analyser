import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Try to import agents (optional)
try:
    from src.agents.portfolio_agent import PortfolioAgent
    AGENT_MODE_AVAILABLE = True
except ImportError:
    AGENT_MODE_AVAILABLE = False

from src.core.portfolio_manager import PortfolioManager

class PortfolioUI:
    """Streamlit UI for portfolio optimization"""
    
    def __init__(self):
        self.portfolio_manager = PortfolioManager()
        self.agent_mode = False
        
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'results' not in st.session_state:
            st.session_state.results = {}
    
    def render_header(self):
        """Render main header and mode selection"""
        st.title("🚀 Portfolio Optimization System")
        st.markdown("### Advanced Multi-Strategy Portfolio Analysis")
        
        # Mode selection
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("📊 Analyze 200+ assets with 200+ signals using advanced ML strategies")
        
        with col2:
            if AGENT_MODE_AVAILABLE:
                self.agent_mode = st.toggle("🤖 Agent Mode", value=False)
                if self.agent_mode:
                    st.success("AI Assistant Active")
                else:
                    st.info("Direct Analysis Mode")
            else:
                st.warning("Agent mode unavailable")
    
    def render_data_upload(self):
        """Data upload and processing section"""
        st.header("📁 Data Processing")
        
        uploaded_file = st.file_uploader(
            "Upload Excel Data (In-sample & Out-sample)", 
            type=['xlsx'], 
            help="Upload Excel file with 4 sheets: In-sample returns, In-sample signals, Out-sample returns, Out-sample signals"
        )
        
        if uploaded_file:
            if st.button("🔄 Load & Process Data", type="primary"):
                with st.spinner("Loading and processing data..."):
                    success, message = self.portfolio_manager.load_data(uploaded_file)
                    
                    if success:
                        st.session_state.data_loaded = True
                        st.success(message)
                        
                        # Show data summary
                        summary = self.portfolio_manager.data_processor.get_data_summary()
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Training Days", summary['training_days'])
                        with col2:
                            st.metric("Assets", summary['num_assets'])
                        with col3:
                            st.metric("Signals", summary['num_signals'])
                        
                        st.json(summary)
                    else:
                        st.error(message)
    
    def render_strategy_execution(self):
        """Strategy execution panel"""
        if not st.session_state.data_loaded:
            st.warning("Please load data first")
            return
        
        st.header("⚡ Strategy Execution")
        
        # Strategy selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Core Strategies")
            
            if st.button("📈 Run Benchmark (OLS)", type="secondary"):
                with st.spinner("Executing benchmark strategy..."):
                    success, message = self.portfolio_manager.run_strategy('benchmark')
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            if st.button("🎯 Run Revised (CCA)", type="secondary"):
                with st.spinner("Executing CCA strategy..."):
                    success, message = self.portfolio_manager.run_strategy('revised_cca')
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        with col2:
            st.subheader("Enhanced Strategies")
            
            # Ridge regression with parameter
            ridge_alpha = st.slider("Ridge α", 0.001, 0.1, 0.01, format="%.3f")
            if st.button("🏔️ Run Ridge Regression", type="secondary"):
                with st.spinner("Executing Ridge strategy..."):
                    success, message = self.portfolio_manager.run_strategy('ridge', alpha=ridge_alpha)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            
            # Lasso regression
            lasso_alpha = st.slider("Lasso α", 0.001, 0.1, 0.01, format="%.3f")
            if st.button("🎪 Run Lasso Regression", type="secondary"):
                with st.spinner("Executing Lasso strategy..."):
                    success, message = self.portfolio_manager.run_strategy('lasso', alpha=lasso_alpha)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Bulk execution
        if st.button("🚀 Execute All Strategies", type="primary"):
            strategies = ['benchmark', 'revised_cca', 'ridge', 'lasso']
            progress_bar = st.progress(0)
            
            for i, strategy in enumerate(strategies):
                with st.spinner(f"Executing {strategy}..."):
                    self.portfolio_manager.run_strategy(strategy)
                    progress_bar.progress((i + 1) / len(strategies))
            
            st.success("All strategies executed!")
    
    def render_results_dashboard(self):
        """Results visualization and comparison"""
        if not self.portfolio_manager.results:
            st.info("No results available. Please execute strategies first.")
            return
        
        st.header("📊 Results Dashboard")
        
        # Strategy comparison table
        comparison = self.portfolio_manager.compare_strategies()
        
        if comparison:
            st.subheader("Strategy Performance Comparison")
            df_comparison = pd.DataFrame(comparison)
            
            # Format numerical columns
            for col in ['In-Sample Sharpe', 'Out-Sample Sharpe', 'Out-Sample Return', 'Out-Sample Volatility']:
                if col in df_comparison.columns:
                    df_comparison[col] = df_comparison[col].round(3)
            
            st.dataframe(df_comparison, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Sharpe ratio comparison
                fig = px.bar(
                    df_comparison, 
                    x='Strategy', 
                    y='Out-Sample Sharpe',
                    title="Out-of-Sample Sharpe Ratio Comparison",
                    color='Out-Sample Sharpe',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk-Return scatter
                fig = px.scatter(
                    df_comparison,
                    x='Out-Sample Volatility',
                    y='Out-Sample Return',
                    size='Out-Sample Sharpe',
                    color='Strategy',
                    title="Risk-Return Profile"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_asset_recommendations(self):
        """Asset recommendation system"""
        if 'revised_cca' not in self.portfolio_manager.results:
            st.info("Please execute CCA strategy first to get recommendations")
            return
        
        st.header("💡 Asset Investment Recommendations")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top Asset Recommendations")
            
            recommendations = self.portfolio_manager.get_asset_recommendation(top_n=15)
            
            if isinstance(recommendations, list):
                rec_df = pd.DataFrame(recommendations)
                rec_df['Expected Return %'] = (rec_df['expected_return'] * 100).round(2)
                
                st.dataframe(
                    rec_df[['rank', 'asset', 'Expected Return %']], 
                    use_container_width=True,
                    hide_index=True
                )
                
                # Visualization
                fig = px.bar(
                    rec_df.head(10),
                    x='asset',
                    y='expected_return',
                    title="Top 10 Assets by Expected Return"
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.error(recommendations)
        
        with col2:
            st.subheader("Portfolio Allocation")
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                ["low", "medium", "high"],
                index=1
            )
            
            if st.button("💼 Generate Portfolio"):
                allocation = self.portfolio_manager.get_portfolio_allocation(risk_tolerance)
                
                if isinstance(allocation, list):
                    alloc_df = pd.DataFrame(allocation)
                    alloc_df['Weight %'] = (alloc_df['weight'] * 100).round(1)
                    
                    st.dataframe(
                        alloc_df[['asset', 'Weight %', 'expected_return']], 
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Pie chart
                    fig = px.pie(
                        alloc_df.head(8),  # Top 8 allocations
                        values='weight',
                        names='asset',
                        title=f"Portfolio Allocation ({risk_tolerance.title()} Risk)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(allocation)
    
    def render_chat_interface(self):
        """Chat interface for agent mode"""
        if not self.agent_mode or not AGENT_MODE_AVAILABLE:
            st.info("Agent mode not available or not enabled")
            return
        
        st.header("💬 AI Portfolio Assistant")
        st.markdown("Ask me anything about your portfolio strategies!")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # User input
        user_input = st.text_input("Ask about your portfolio:", placeholder="e.g., Why does CCA outperform OLS?")
        
        if st.button("Send") and user_input:
            # This would integrate with LangChain agents when available
            response = f"I understand you're asking: '{user_input}'. Agent functionality will be available when LLM is connected."
            
            st.session_state.chat_history.append(("User", user_input))
            st.session_state.chat_history.append(("AI", response))
        
        # Display chat history
        for sender, message in st.session_state.chat_history[-10:]:  # Last 10 messages
            if sender == "User":
                st.write(f"🗣️ **You:** {message}")
            else:
                st.write(f"🤖 **AI:** {message}")
    
    def run(self):
        """Main UI execution"""
        self.render_header()
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Data", "⚡ Strategies", "📊 Results", 
            "💡 Recommendations", "💬 Chat"
        ])
        
        with tab1:
            self.render_data_upload()
        
        with tab2:
            self.render_strategy_execution()
        
        with tab3:
            self.render_results_dashboard()
        
        with tab4:
            self.render_asset_recommendations()
        
        with tab5:
            self.render_chat_interface()