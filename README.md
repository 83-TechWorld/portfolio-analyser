# Portfolio Optimizer

Advanced portfolio optimization system using machine learning strategies and interactive visualization.

## Features

- Multiple portfolio optimization strategies:
  - Benchmark OLS (Ordinary Least Squares)
  - Revised CCA (Canonical Correlation Analysis)
  - Ridge Regression
  - Lasso Regression
- Interactive Streamlit UI
- Real-time performance visualization
- Asset recommendations
- Risk-adjusted portfolio allocation
- Optional AI assistant mode (requires additional setup)

## Setup

1. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Optional: Install LLM dependencies for AI assistant mode:
```bash
pip install langchain openai
```

## Usage

1. Run the application:
```bash
streamlit run main.py
```

2. Upload your data file:
   - Excel file with 4 sheets:
     - In-sample returns
     - In-sample signals
     - Out-sample returns
     - Out-sample signals
   - Each sheet should have a 'Day' column and asset/signal columns

3. Execute strategies:
   - Try different strategies individually
   - Or run all strategies at once
   - Compare performance metrics

4. View recommendations:
   - Get top asset recommendations
   - Generate risk-adjusted portfolio allocation
   - View performance visualizations

## Data Format

Your Excel file should follow this structure:
- Each sheet has a 'Day' column as the first column
- Returns sheets: columns are asset returns
- Signals sheets: columns are trading signals
- All sheets must have matching day indices

## Project Structure

```
portfolio-analyzer/
├── src/
│   ├── core/           # Core implementation
│   │   ├── data_processor.py
│   │   ├── strategies.py
│   │   ├── evaluator.py
│   │   └── portfolio_manager.py
│   ├── ui/            # User interface
│   │   └── streamlit_app.py
│   └── agents/        # Optional LLM integration
│       └── portfolio_agent.py
├── data/              # Data directory
│   ├── raw/          # Original data files
│   └── processed/    # Processed data
├── notebooks/        # Analysis notebooks
├── main.py          # Application entry
└── requirements.txt  # Dependencies
```

## Contributing

Feel free to contribute by:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
