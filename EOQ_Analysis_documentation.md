# Economic Order Quantity (EOQ) Analysis Documentation

## Methodology

### 1. Theoretical Foundation
1. **Economic Order Quantity Model**
   - Classical Wilson EOQ formula:
     ```
     EOQ = √((2 × D × S) / H)
     
     Where:
     D = Annual demand
     S = Fixed cost per order
     H = Annual holding cost per unit
     ```
   - Assumptions:
     * Constant and known demand
     * Fixed ordering cost
     * Fixed holding cost
     * No stockouts or shortages
     * Instantaneous delivery

2. **Total Cost Components**
   ```
   Total Annual Cost = Ordering Cost + Holding Cost
   Where:
   Ordering Cost = (D/Q) × S
   Holding Cost = (Q/2) × H
   Q = Order quantity
   ```

### 2. Data Preprocessing
1. **Annual Demand Calculation**
   ```python
   item_analysis['Days_Analyzed'] = item_analysis['Transaction Date'].dt.days
   item_analysis['Annual_Demand'] = (
       item_analysis['Quantity'] * 365 / 
       item_analysis['Days_Analyzed']
   )
   ```

2. **Cost Parameter Normalization**
   - Holding cost calculation:
     ```python
     holding_cost = unit_cost * holding_cost_percentage
     ```
   - Components included:
     * Storage costs
     * Insurance
     * Depreciation
     * Opportunity cost of capital

### 3. EOQ Calculation Process
1. **Basic EOQ Computation**
   ```python
   def calculate_optimal_order_quantity(self, 
                                     holding_cost_percentage: float = 0.20,
                                     ordering_cost: float = 50.0):
       eoq = np.sqrt(
           (2 * annual_demand * ordering_cost) / 
           (unit_cost * holding_cost_percentage)
       )
   ```

2. **Cost Optimization**
   ```python
   # Current costs
   current_orders_per_year = order_count * 365 / days_analyzed
   current_avg_order_size = annual_demand / current_orders_per_year
   
   # Optimal costs
   optimal_orders_per_year = annual_demand / eoq
   
   # Cost calculations
   current_annual_cost = (
       current_orders_per_year * ordering_cost +
       (current_avg_order_size / 2) * holding_cost
   )
   
   optimal_annual_cost = (
       optimal_orders_per_year * ordering_cost +
       (eoq / 2) * holding_cost
   )
   ```

### 4. Analysis Components

1. **Item Classification**
   - By value (ABC analysis):
     * A: Top 20% by value
     * B: Next 30% by value
     * C: Remaining 50% by value
   - By order frequency:
     * High frequency (>12 orders/year)
     * Medium frequency (4-12 orders/year)
     * Low frequency (<4 orders/year)

2. **Cost Sensitivity Analysis**
   - Impact of parameter changes:
     * Holding cost variations
     * Ordering cost variations
     * Demand fluctuations
   - Scenario modeling:
     * Best case
     * Worst case
     * Most likely case

3. **Safety Stock Considerations**
   ```python
   safety_stock = z_score * std_dev_demand * sqrt(lead_time)
   reorder_point = (avg_daily_demand * lead_time) + safety_stock
   ```

### 5. Validation and Adjustment

1. **Data Quality Checks**
   ```python
   def validate_input_data(self):
       # Ensure numeric columns
       self.df['Quantity'] = pd.to_numeric(
           self.df['Quantity'], 
           errors='coerce'
       )
       self.df['SupplierUnit Price'] = pd.to_numeric(
           self.df['SupplierUnit Price'], 
           errors='coerce'
       )
   ```

2. **Business Constraint Integration**
   - Minimum order quantities
   - Storage capacity limits
   - Budget constraints
   - Supplier lead times

3. **Result Validation**
   - Historical comparison
   - Industry benchmarks
   - Practical feasibility checks

## Implementation Details

### 1. Core Analysis Class
```python
class CostOptimizationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for cost optimization analysis"""
        try:
            # Ensure numeric columns
            self.df['Quantity'] = pd.to_numeric(
                self.df['Quantity'], 
                errors='coerce'
            )
            self.df['SupplierUnit Price'] = pd.to_numeric(
                self.df['SupplierUnit Price'], 
                errors='coerce'
            )
            
            # Calculate total value
            self.df['Total Value'] = (
                self.df['Quantity'] * 
                self.df['SupplierUnit Price']
            )
            
            # Add timeframe analysis
            self.df['Transaction Date'] = pd.to_datetime(
                self.df['Transaction Date']
            )
            self.df['Year_Month'] = self.df['Transaction Date'].dt.strftime(
                '%Y-%m'
            )
            
            # Calculate frequency of orders
            self.df['Order_Count'] = 1
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
```

### 2. Output Structure
```python
{
    'eoq_analysis': [
        {
            'category': str,
            'supplier': str,
            'annual_demand': float,
            'unit_cost': float,
            'eoq': float,
            'current_order_size': float,
            'optimal_orders_per_year': float,
            'current_annual_cost': float,
            'optimal_annual_cost': float,
            'potential_savings': float,
            'savings_percentage': float
        }
    ],
    'metrics': {
        'total_potential_savings': float,
        'average_savings_percentage': float,
        'items_analyzed': int
    }
}
```

## Best Practices

### 1. Implementation Guidelines
1. **Phased Approach**
   - Start with high-value items
   - Validate results with small pilot
   - Gradually expand implementation

2. **Regular Review and Adjustment**
   - Monthly parameter review
   - Quarterly performance assessment
   - Annual strategy alignment

### 2. Parameter Selection
1. **Holding Cost Determination**
   - Storage space costs
   - Insurance costs
   - Handling costs
   - Opportunity cost of capital
   - Obsolescence risk

2. **Ordering Cost Components**
   - Purchase order processing
   - Setup costs
   - Quality inspection
   - Receiving and handling
   - Administrative overhead

## Integration and Visualization

### 1. Dashboard Components
1. **Summary Metrics**
   - Total potential savings
   - Average savings percentage
   - Number of items analyzed

2. **Interactive Visualizations**
   - Current vs. Optimal Order Size comparison
   - Cost breakdown analysis
   - Savings opportunity distribution

### 2. Export Capabilities
1. **Report Formats**
   - Detailed EOQ analysis
   - Cost comparison summary
   - Implementation recommendations

## Limitations and Considerations

### 1. Model Assumptions
- Constant demand rate
- Fixed ordering and holding costs
- No quantity discounts
- Instant replenishment
- No shortages allowed

### 2. Practical Constraints
- Storage capacity limits
- Budget constraints
- Supplier capacity
- Lead time variations

### 3. Data Requirements
- Minimum 6 months of transaction history
- Accurate cost data
- Reliable demand patterns

## Future Enhancements

### 1. Advanced Analytics
- Machine learning for demand forecasting
- Multi-item optimization
- Dynamic parameter adjustment

### 2. Integration Capabilities
- Real-time data processing
- API connectivity
- ERP system integration

### 3. Additional Features
- Variable cost modeling
- Seasonal demand adjustment
- Multi-warehouse optimization
- Supplier constraint integration

## Error Handling

### 1. Data Validation
```python
def validate_parameters(self, holding_cost_percentage, ordering_cost):
    if holding_cost_percentage <= 0:
        raise ValueError("Holding cost percentage must be positive")
    if ordering_cost <= 0:
        raise ValueError("Ordering cost must be positive")
```

### 2. Exception Management
```python
try:
    eoq_results = self.calculate_optimal_order_quantity()
except Exception as e:
    log_error(f"EOQ calculation failed: {str(e)}")
    return None
```

## Performance Metrics

### 1. Analysis Performance
- Calculation time
- Memory usage
- Error rates

### 2. Business Impact
- Actual vs. predicted savings
- Implementation success rate
- Inventory turnover improvement
- Working capital reduction