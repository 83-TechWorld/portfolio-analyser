from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI  # or other LLM

class PortfolioAgent:
    def __init__(self, portfolio_manager):
        self.portfolio_manager = portfolio_manager
        self.memory = ConversationBufferMemory()
        
        # Define tools for the agent
        self.tools = [
            Tool(
                name="run_strategy",
                func=self._run_strategy_tool,
                description="Execute a portfolio strategy"
            ),
            Tool(
                name="get_recommendations",
                func=self._get_recommendations_tool,
                description="Get asset investment recommendations"
            ),
            Tool(
                name="compare_strategies",
                func=self._compare_strategies_tool,
                description="Compare strategy performance"
            )
        ]
    
    def _run_strategy_tool(self, strategy_name):
        success, message = self.portfolio_manager.run_strategy(strategy_name)
        return message
    
    def _get_recommendations_tool(self, query):
        recommendations = self.portfolio_manager.get_asset_recommendation()
        return str(recommendations)
    
    def _compare_strategies_tool(self, query):
        comparison = self.portfolio_manager.compare_strategies()
        return str(comparison)