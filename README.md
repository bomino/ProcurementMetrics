# Procurement Analysis Dashboard

## Overview
The **Procurement Analysis Dashboard** is an interactive data analysis tool built using **Streamlit** that provides comprehensive insights into procurement data. It helps procurement teams and analysts understand key metrics such as spend distribution, supplier performance, cost optimization, and project analysis. This dashboard facilitates data-driven decision-making by presenting detailed visualizations and metrics, enabling better procurement strategies and optimizations.

## Main Features
- **Spend Analysis**: Detailed breakdown of monthly and yearly spend trends, spend by supplier, category, and project.
- **Supplier Performance**: Analysis of average lead times, on-time delivery rates, price statistics, and total savings.
- **Category Insights**: Insights into category volumes, total savings, and price trends.
- **Project Performance**: Evaluation of project spending, on-time delivery rates, and material cost distribution.
- **Cost Optimization**: Identification of bundle purchase opportunities and calculation of optimal order quantities (EOQ) with potential savings.
- **Downloadable Reports**: Ability to download consolidated analysis in Excel format.

## Project Structure
The project is implemented as a Python-based web application using Streamlit. The main code resides in `app.py`, with helper classes for analysis:
- **ProcurementAnalyzer**: Handles the preparation and analysis of procurement data.
- **CostOptimizationAnalyzer**: Provides cost optimization insights, including EOQ and bundle opportunities.

## Setup Instructions
### Prerequisites
- Python 3.7 or higher
- Recommended: Virtual environment (e.g., `venv`, `conda`)

### Installation Steps
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the application**:
   ```bash
   streamlit run app.py
   ```
5. **Upload your data**: Use the sidebar file uploader in the application to upload a CSV file containing your procurement data.

### Required Data Format
Ensure your CSV file includes the following columns:
- `Transaction Date`
- `Supplier`
- `Category`
- `Project`
- `Quantity`
- `SupplierUnit Price`
- `CostSavings`
- `Delivery Status`

### Example Data Schema
| Transaction Date | Supplier       | Category | Project    | Quantity | SupplierUnit Price | CostSavings | Delivery Status |
|------------------|----------------|----------|------------|----------|---------------------|-------------|------------------|
| 2024-01-15       | ABC Supplies   | Hardware | Project X  | 50       | 25.00               | 100.00      | On Time          |
| 2024-02-20       | XYZ Traders    | Software | Project Y  | 10       | 500.00              | 0.00        | Late             |

## Usage Examples
### Spend Analysis
Navigate to the **Spend Analysis** tab to view:
- **Monthly Spend Trend**: A line chart depicting spending over time.
- **Yearly Spend Pattern**: A bar chart for total yearly spend.
- **Spend by Supplier and Category**: Visuals showing distribution by supplier and category.

### Supplier Performance
- **Average Lead Time by Supplier**: Bar chart ranking suppliers based on their average lead time.
- **On-Time Delivery Rate**: Performance chart highlighting supplier reliability.
- **Price Distribution**: Box plots showing unit price variations.

### Cost Optimization
- **Bundle Opportunities**: Identify items frequently bought together with potential savings.
- **EOQ Analysis**: Visuals and tables outlining optimal order quantities for cost savings.

## Additional Features
- **Downloadable Reports**: Generate and download Excel reports of your analysis from the sidebar.
- **Manual Threshold Adjustment**: Override default threshold settings for bundle identification and optimization.

## Troubleshooting
- Ensure the CSV file has all required columns.
- Check that date columns are properly formatted (`YYYY-MM-DD`).
- If you encounter errors, refer to the detailed error messages in the application for guidance.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
For questions or support, please contact M.Lawali at mlawali@qidaya.com

