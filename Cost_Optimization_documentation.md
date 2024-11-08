# Cost Optimization Methodology Documentation

## 1. Bundle Purchase Opportunities Identification

### Overview
The bundle purchase opportunities identification system uses advanced statistical analysis and pattern recognition to identify items that could be purchased together for potential cost savings. The method combines several analytical approaches to ensure robust and actionable recommendations.

### Methodology

#### 1.1 Purchase Pattern Analysis
The system analyzes purchase patterns through the following steps:

a) **Transaction Grouping**
```python
- Groups purchases by transaction date
- Identifies items purchased in the same transaction
- Creates a purchase frequency matrix
```

b) **Co-occurrence Analysis**
```python
- Calculates how often items are purchased together
- Identifies regular purchasing patterns
- Establishes baseline frequency thresholds
```

#### 1.2 Correlation Analysis

a) **Purchase Pattern Correlation**
- Creates a cross-tabulation matrix of items and purchase dates
- Calculates correlation coefficients between item pairs
- Uses the formula:
```
correlation_coefficient = 
    covariance(item1_purchases, item2_purchases) / 
    (std_dev(item1_purchases) * std_dev(item2_purchases))
```

b) **Correlation Thresholds**
- Default minimum correlation: 0.6 (configurable)
- Correlation interpretation:
  * 0.6-0.7: Moderate correlation
  * 0.7-0.8: Strong correlation
  * 0.8+: Very strong correlation

#### 1.3 Bundle Opportunity Scoring

The system scores bundle opportunities based on multiple factors:

a) **Primary Factors**
```
- Purchase correlation coefficient
- Co-purchase frequency
- Total transaction value
- Historical purchase patterns
```

b) **Saving Calculation**
```python
potential_savings = (item1_avg_value + item2_avg_value) * discount_factor
where:
- item1_avg_value = average transaction value for first item
- item2_avg_value = average transaction value for second item
- discount_factor = estimated bulk purchase discount (default: 10%)
```

#### 1.4 Filtering and Ranking

Bundle opportunities are filtered and ranked based on:

a) **Minimum Thresholds**
```
- Correlation coefficient ≥ minimum_correlation
- Co-purchase frequency ≥ minimum_frequency
- Potential savings > 0
```

b) **Ranking Criteria**
```
1. Potential savings amount
2. Correlation strength
3. Purchase frequency
4. Supplier consistency
```

## 2. Optimal Order Quantity Calculator

### Overview
The Optimal Order Quantity Calculator implements the Economic Order Quantity (EOQ) model with additional refinements for practical application in procurement scenarios.

### Methodology

#### 2.1 Economic Order Quantity (EOQ) Model

a) **Basic EOQ Formula**
```
EOQ = √((2 * D * S) / H)
where:
D = Annual demand
S = Ordering cost per order
H = Annual holding cost per unit
```

b) **Input Parameters**
```python
1. Annual Demand (D)
   - Calculated from historical data
   - Extrapolated for partial year data
   - Adjusted for seasonality

2. Ordering Cost (S)
   - Fixed cost per order
   - Includes processing, shipping, handling
   - Default: $50 (configurable)

3. Holding Cost (H)
   - Calculated as percentage of unit cost
   - Default: 20% of unit cost annually
   - Includes storage, insurance, capital cost
```

#### 2.2 Cost Calculations

a) **Current Costs**
```python
current_annual_cost = 
    (current_orders_per_year * ordering_cost) +
    (current_avg_order_size / 2 * holding_cost)
```

b) **Optimal Costs**
```python
optimal_annual_cost = 
    (optimal_orders_per_year * ordering_cost) +
    (eoq / 2 * holding_cost)
```

c) **Savings Calculation**
```python
potential_savings = current_annual_cost - optimal_annual_cost
savings_percentage = (potential_savings / current_annual_cost) * 100
```

#### 2.3 Analysis Refinements

a) **Time Period Normalization**
```python
annual_demand = actual_demand * (365 / days_analyzed)
```

b) **Order Pattern Analysis**
```python
current_orders_per_year = order_count * (365 / days_analyzed)
current_avg_order_size = annual_demand / current_orders_per_year
```

#### 2.4 Implementation Considerations

a) **Data Requirements**
```
- Transaction history
- Order quantities
- Unit prices
- Order dates
- Supplier information
```

b) **Adjustments for:**
```
- Lead time variations
- Minimum order quantities
- Quantity discounts
- Storage constraints
```

### Practical Application

#### 1. Bundle Purchase Implementation
```
1. Review top bundle opportunities
2. Validate correlation strength
3. Check supplier capabilities
4. Calculate potential savings
5. Implement pilot programs
6. Monitor performance
```

#### 2. EOQ Implementation
```
1. Start with highest-saving items
2. Adjust for practical constraints
3. Implement gradually
4. Monitor inventory levels
5. Track actual savings
6. Refine parameters based on results
```

### Performance Metrics

#### Bundle Purchase Metrics
```
1. Savings achieved vs. predicted
2. Bundle adoption rate
3. Supplier compliance
4. Order consolidation rate
```

#### EOQ Metrics
```
1. Actual vs. predicted savings
2. Order frequency adherence
3. Stock-out incidents
4. Inventory carrying costs
5. Order processing costs
```

This documentation provides a comprehensive overview of the methodologies used in both cost optimization features. The system is designed to be both theoretically sound and practically applicable, with configurable parameters to adapt to different procurement scenarios and business requirements.