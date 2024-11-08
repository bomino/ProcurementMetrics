import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import json
import io  # Added missing import




class ProcurementAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analysis"""
        try:
            # Add time-based columns
            self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
            self.df['Year'] = self.df['Transaction Date'].dt.year
            self.df['Month'] = self.df['Transaction Date'].dt.month
            self.df['Quarter'] = self.df['Transaction Date'].dt.quarter
            self.df['Month_Year'] = self.df['Transaction Date'].dt.strftime('%Y-%m')
            
            # Handle missing values before calculations
            self.df['Quantity'] = self.df['Quantity'].fillna(0)
            self.df['SupplierUnit Price'] = self.df['SupplierUnit Price'].fillna(0)
            self.df['CostSavings'] = self.df['CostSavings'].fillna(0)
            
            # Calculate additional metrics
            self.df['Total Value'] = self.df['Quantity'] * self.df['SupplierUnit Price']
            self.df['Savings'] = self.df['CostSavings']
            self.df['Is_Late'] = self.df['Delivery Status'] == 'Late'
            
            # Add data validation
            if self.df['Total Value'].isnull().any():
                st.warning("Some total values could not be calculated. Please check Quantity and SupplierUnit Price data.")
                
        except Exception as e:
            st.error(f"Error in data preparation: {str(e)}")
            print(f"Data preparation error details: {e}")

# In the ProcurementAnalyzer class, update the analyze_spend method:

    # Update the project_savings calculation in the analyze_spend method
    def analyze_spend(self):
        """Analyze spending patterns"""
        try:
            # Time-based spend analysis
            self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
            
            monthly_data = (
                self.df.groupby(self.df['Transaction Date'].dt.strftime('%Y-%m'))['Total Value']
                .sum()
                .reset_index()
            )
            monthly_data.columns = ['Month_Year', 'Total Value']
            
            yearly_spend = (
                self.df.groupby(self.df['Transaction Date'].dt.year)['Total Value']
                .sum()
            )
            
            category_spend = self.df.groupby('Category')['Total Value'].sum()
            supplier_spend = self.df.groupby('Supplier')['Total Value'].sum()
            project_spend = self.df.groupby('Project')['Total Value'].sum()
            
            # Update: Use transform instead of fillna in groupby
            self.df['CostSavings_filled'] = self.df.groupby('Project')['CostSavings'].transform(lambda x: x.fillna(0))
            project_savings = self.df.groupby('Project')['CostSavings_filled'].sum()
            
            return {
                'monthly_spend': dict(zip(monthly_data['Month_Year'], monthly_data['Total Value'])),
                'yearly_spend': yearly_spend.to_dict(),
                'category_spend': category_spend.to_dict(),
                'supplier_spend': supplier_spend.to_dict(),
                'project_spend': project_spend.to_dict(),
                'project_savings': project_savings.to_dict()
            }
        except Exception as e:
            st.error(f"Error in spend analysis: {str(e)}")
            print(f"Detailed error: {e}")  # For debugging
            return None

    # Update the analyze_supplier_performance method to handle the plotly warning
    def analyze_supplier_performance(self):
        """Analyze supplier performance metrics"""
        try:
            # Lead time analysis
            avg_lead_time = self.df.groupby('Supplier')['Lead Time (Days)'].mean()

            # Delivery performance
            delivery_stats = (
                self.df.groupby('Supplier')
                .agg({
                    'Is_Late': ['count', 'sum']
                })
            )
            delivery_stats.columns = ['total_deliveries', 'late_deliveries']
            delivery_stats['on_time_rate'] = ((delivery_stats['total_deliveries'] - delivery_stats['late_deliveries']) / 
                                            delivery_stats['total_deliveries'] * 100)
            
            # Price analysis with proper transform handling
            self.df['CostSavings_filled'] = self.df.groupby('Supplier')['CostSavings'].transform(lambda x: x.fillna(0))
            price_stats = self.df.groupby('Supplier').agg({
                'SupplierUnit Price': ['mean', 'std'],
                'CostSavings_filled': 'sum'
            })
            price_stats.columns = ['avg_price', 'price_std', 'total_savings']
            
            # Price distribution - Keep supplier names as strings for Plotly compatibility
            price_distribution = {}
            for supplier in self.df['Supplier'].unique():
                supplier_data = self.df[self.df['Supplier'] == supplier]
                price_distribution[str(supplier)] = supplier_data['SupplierUnit Price'].dropna().tolist()

            return {
                'avg_lead_time': avg_lead_time.to_dict(),
                'delivery_performance': {
                    'total_deliveries': delivery_stats['total_deliveries'].to_dict(),
                    'late_deliveries': delivery_stats['late_deliveries'].to_dict(),
                    'on_time_rate': delivery_stats['on_time_rate'].to_dict()
                },
                'price_stats': {
                    'avg_price': price_stats['avg_price'].to_dict(),
                    'price_std': price_stats['price_std'].to_dict(),
                    'total_savings': price_stats['total_savings'].to_dict()
                },
                'price_distribution': price_distribution
            }
        except Exception as e:
            st.error(f"Error in supplier analysis: {str(e)}")
            return None

    def analyze_project_performance(self):
        """Analyze project performance metrics"""
        try:
            # Project spend analysis
            project_spend = self.df.groupby('Project')['Total Value'].sum()
            projects = sorted(list(project_spend.index))  # Get sorted list of unique projects

            # Project delivery performance
            delivery_stats = (
                self.df.groupby('Project')
                .agg({
                    'Is_Late': ['count', 'sum']
                })
            )
            delivery_stats.columns = ['total_deliveries', 'late_deliveries']
            delivery_stats['on_time_rate'] = (
                (delivery_stats['total_deliveries'] - delivery_stats['late_deliveries']) / 
                delivery_stats['total_deliveries'].replace(0, np.nan) * 100
            ).fillna(0)
            
            # Material analysis
            material_analysis = (
                self.df.groupby(['Project', 'Category'])
                .agg({
                    'Quantity': 'sum',
                    'Total Value': 'sum'
                })
                .reset_index()
            )

            # Ensure all projects are represented in delivery_stats
            delivery_stats = delivery_stats.reindex(projects, fill_value=0)

            # Create consistent dictionaries for all projects
            project_data = {
                'project_spend': {},
                'delivery_performance': {
                    'total_deliveries': {},
                    'late_deliveries': {},
                    'on_time_rate': {}
                },
                'material_analysis': {
                    'quantities': {},
                    'values': {},
                    'category_breakdown': material_analysis.to_dict('records')
                }
            }

            # Fill in data for each project
            for project in projects:
                project_data['project_spend'][project] = float(project_spend.get(project, 0))
                project_data['delivery_performance']['total_deliveries'][project] = int(delivery_stats.loc[project, 'total_deliveries'])
                project_data['delivery_performance']['late_deliveries'][project] = int(delivery_stats.loc[project, 'late_deliveries'])
                project_data['delivery_performance']['on_time_rate'][project] = float(delivery_stats.loc[project, 'on_time_rate'])
                
                # Get quantities and values from material_analysis
                project_quantities = material_analysis[material_analysis['Project'] == project]['Quantity'].sum()
                project_values = material_analysis[material_analysis['Project'] == project]['Total Value'].sum()
                project_data['material_analysis']['quantities'][project] = float(project_quantities)
                project_data['material_analysis']['values'][project] = float(project_values)

            return project_data

        except Exception as e:
            st.error(f"Error in project analysis: {str(e)}")
            print(f"Detailed project analysis error: {e}")  # For debugging
            return None

    def analyze_category_insights(self):
        """Analyze category and commodity trends"""
        try:
            # Category analysis with consistent data structures
            categories = sorted(self.df['Category'].unique())
            commodities = sorted(self.df['Commodity'].unique())
            
            category_data = {
                'category_volume': {},
                'category_savings': {},
                'commodity_spend': {},
                'commodity_volume': {},
                'price_trends': {}
            }
            
            # Calculate metrics for each category
            for category in categories:
                category_mask = self.df['Category'] == category
                category_data['category_volume'][category] = float(self.df[category_mask]['Quantity'].sum())
                category_data['category_savings'][category] = float(self.df[category_mask]['CostSavings'].sum())
            
            # Calculate metrics for each commodity
            for commodity in commodities:
                commodity_mask = self.df['Commodity'] == commodity
                category_data['commodity_spend'][commodity] = float(self.df[commodity_mask]['Total Value'].sum())
                category_data['commodity_volume'][commodity] = float(self.df[commodity_mask]['Quantity'].sum())
            
            # Calculate price trends
            price_trends = (
                self.df.groupby(['Month_Year', 'Category'])['SupplierUnit Price']
                .mean()
                .round(2)
            )
            
            # Convert price trends to dictionary with consistent format
            for (month_year, category) in price_trends.index:
                key = f"{month_year}_{category}"
                category_data['price_trends'][key] = float(price_trends.loc[(month_year, category)])
            
            return category_data

        except Exception as e:
            st.error(f"Error in category analysis: {str(e)}")
            print(f"Detailed category analysis error: {e}")  # For debugging
            return None



import pandas as pd
import numpy as np
from scipy.stats import zscore
from typing import Dict, List, Tuple

class CostOptimizationAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for cost optimization analysis"""
        try:
            # Ensure numeric columns
            self.df['Quantity'] = pd.to_numeric(self.df['Quantity'], errors='coerce')
            self.df['SupplierUnit Price'] = pd.to_numeric(self.df['SupplierUnit Price'], errors='coerce')
            
            # Calculate total value
            self.df['Total Value'] = self.df['Quantity'] * self.df['SupplierUnit Price']
            
            # Add timeframe analysis
            self.df['Transaction Date'] = pd.to_datetime(self.df['Transaction Date'])
            self.df['Year_Month'] = self.df['Transaction Date'].dt.strftime('%Y-%m')
            
            # Calculate frequency of orders
            self.df['Order_Count'] = 1
            
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")

    def identify_bundle_opportunities(self, 
                                   min_correlation: float = 0.6,
                                   min_frequency: int = 3) -> Dict:
        """
        Identify potential bundle purchase opportunities based on:
        1. Items frequently purchased together
        2. Items with correlated purchase patterns
        3. Items from the same supplier with similar order patterns
        
        Parameters:
        - min_correlation: Minimum correlation coefficient to consider items related
        - min_frequency: Minimum number of times items should be purchased together
        
        Returns:
        - Dictionary containing bundle opportunities and their potential savings
        """
        try:
            # Group items by transaction to find co-purchases
            transaction_groups = self.df.groupby(['Transaction Date', 'Category', 'Supplier'])
            
            # Analyze purchase patterns
            purchase_patterns = pd.crosstab(
                self.df['Transaction Date'],
                [self.df['Category'], self.df['Supplier']]
            )
            
            # Calculate correlations between items
            correlations = purchase_patterns.corr()
            
            # Find strongly correlated items
            bundle_opportunities = []
            
            for i in range(len(correlations.columns)):
                for j in range(i + 1, len(correlations.columns)):
                    if correlations.iloc[i, j] >= min_correlation:
                        item1 = correlations.columns[i]
                        item2 = correlations.columns[j]
                        
                        # Check frequency of co-occurrence
                        co_occurrences = len(purchase_patterns[
                            (purchase_patterns[item1] > 0) & 
                            (purchase_patterns[item2] > 0)
                        ])
                        
                        if co_occurrences >= min_frequency:
                            # Calculate potential savings (assuming 10% bundle discount)
                            item1_avg_value = self.df[
                                (self.df['Category'] == item1[0]) & 
                                (self.df['Supplier'] == item1[1])
                            ]['Total Value'].mean()
                            
                            item2_avg_value = self.df[
                                (self.df['Category'] == item2[0]) & 
                                (self.df['Supplier'] == item2[1])
                            ]['Total Value'].mean()
                            
                            potential_savings = (item1_avg_value + item2_avg_value) * 0.10
                            
                            bundle_opportunities.append({
                                'item1_category': item1[0],
                                'item1_supplier': item1[1],
                                'item2_category': item2[0],
                                'item2_supplier': item2[1],
                                'correlation': correlations.iloc[i, j],
                                'co_occurrences': co_occurrences,
                                'potential_savings': potential_savings
                            })
            
            # Sort bundle opportunities by potential savings
            bundle_opportunities.sort(key=lambda x: x['potential_savings'], reverse=True)
            
            # Calculate additional metrics
            total_potential_savings = sum(b['potential_savings'] for b in bundle_opportunities)
            avg_correlation = np.mean([b['correlation'] for b in bundle_opportunities])
            
            return {
                'bundle_opportunities': bundle_opportunities,
                'metrics': {
                    'total_potential_savings': total_potential_savings,
                    'number_of_opportunities': len(bundle_opportunities),
                    'average_correlation': avg_correlation
                }
            }
            
        except Exception as e:
            print(f"Error in bundle identification: {str(e)}")
            return None

    def calculate_optimal_order_quantity(self, 
                                      holding_cost_percentage: float = 0.20,
                                      ordering_cost: float = 50.0) -> Dict:
        """
        Calculate the Economic Order Quantity (EOQ) for each item.
        
        Parameters:
        - holding_cost_percentage: Annual holding cost as a percentage of unit cost
        - ordering_cost: Fixed cost per order
        
        Returns:
        - Dictionary containing EOQ calculations and recommendations
        """
        try:
            # Group by Category and Supplier to analyze each unique item
            item_analysis = self.df.groupby(['Category', 'Supplier']).agg({
                'Quantity': 'sum',
                'SupplierUnit Price': 'mean',
                'Order_Count': 'sum',
                'Transaction Date': lambda x: x.max() - x.min()
            }).reset_index()
            
            # Calculate annual demand (extrapolate if needed)
            item_analysis['Days_Analyzed'] = item_analysis['Transaction Date'].dt.days
            item_analysis['Annual_Demand'] = (item_analysis['Quantity'] * 365 / 
                                            item_analysis['Days_Analyzed'])
            
            # Calculate EOQ and related metrics
            eoq_analysis = []
            
            for _, row in item_analysis.iterrows():
                # Basic EOQ calculation
                annual_demand = row['Annual_Demand']
                unit_cost = row['SupplierUnit Price']
                holding_cost = unit_cost * holding_cost_percentage
                
                eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
                
                # Calculate current vs optimal costs
                current_orders_per_year = row['Order_Count'] * 365 / row['Days_Analyzed']
                current_avg_order_size = annual_demand / current_orders_per_year
                
                optimal_orders_per_year = annual_demand / eoq
                
                # Calculate costs
                current_annual_cost = (
                    current_orders_per_year * ordering_cost +
                    (current_avg_order_size / 2) * holding_cost
                )
                
                optimal_annual_cost = (
                    optimal_orders_per_year * ordering_cost +
                    (eoq / 2) * holding_cost
                )
                
                potential_savings = current_annual_cost - optimal_annual_cost
                
                eoq_analysis.append({
                    'category': row['Category'],
                    'supplier': row['Supplier'],
                    'annual_demand': annual_demand,
                    'unit_cost': unit_cost,
                    'eoq': eoq,
                    'current_order_size': current_avg_order_size,
                    'optimal_orders_per_year': optimal_orders_per_year,
                    'current_annual_cost': current_annual_cost,
                    'optimal_annual_cost': optimal_annual_cost,
                    'potential_savings': potential_savings,
                    'savings_percentage': (potential_savings / current_annual_cost) * 100
                })
            
            # Sort by potential savings
            eoq_analysis.sort(key=lambda x: x['potential_savings'], reverse=True)
            
            # Calculate summary metrics
            total_potential_savings = sum(item['potential_savings'] for item in eoq_analysis)
            avg_savings_percentage = np.mean([item['savings_percentage'] for item in eoq_analysis])
            
            return {
                'eoq_analysis': eoq_analysis,
                'metrics': {
                    'total_potential_savings': total_potential_savings,
                    'average_savings_percentage': avg_savings_percentage,
                    'items_analyzed': len(eoq_analysis)
                }
            }
            
        except Exception as e:
            print(f"Error in EOQ calculation: {str(e)}")
            return None
        

    def optimize_bundle_thresholds(self, 
                             business_size: str = 'medium',
                             industry: str = 'general',
                             risk_tolerance: str = 'medium',
                             transaction_volume: str = 'medium',
                             seasonality: str = 'low') -> Dict[str, float]:
        """
        Optimize bundle thresholds based on business context and requirements.
        
        Parameters:
        - business_size: 'small', 'medium', or 'large'
        - industry: 'manufacturing', 'retail', 'services', or 'general'
        - risk_tolerance: 'low', 'medium', or 'high'
        - transaction_volume: 'low', 'medium', or 'high'
        - seasonality: 'low', 'medium', or 'high'
        
        Returns:
        - Dictionary with optimized correlation and frequency thresholds
        """
        try:
            # Base thresholds
            base_thresholds = {
                'correlation': 0.60,
                'frequency': 3
            }
            
            # Business size adjustments
            size_adjustments = {
                'small': {'correlation': -0.10, 'frequency': -1},
                'medium': {'correlation': 0, 'frequency': 0},
                'large': {'correlation': 0.10, 'frequency': 1}
            }
            
            # Industry adjustments
            industry_adjustments = {
                'manufacturing': {'correlation': 0.10, 'frequency': 1},
                'retail': {'correlation': -0.05, 'frequency': 0},
                'services': {'correlation': -0.10, 'frequency': -1},
                'general': {'correlation': 0, 'frequency': 0}
            }
            
            # Risk tolerance adjustments
            risk_adjustments = {
                'low': {'correlation': 0.15, 'frequency': 2},
                'medium': {'correlation': 0, 'frequency': 0},
                'high': {'correlation': -0.15, 'frequency': -1}
            }
            
            # Volume adjustments
            volume_adjustments = {
                'low': {'correlation': -0.05, 'frequency': -1},
                'medium': {'correlation': 0, 'frequency': 0},
                'high': {'correlation': 0.05, 'frequency': 1}
            }
            
            # Seasonality adjustments
            seasonality_adjustments = {
                'low': {'correlation': 0, 'frequency': 0},
                'medium': {'correlation': -0.05, 'frequency': 0},
                'high': {'correlation': -0.10, 'frequency': 1}
            }
            
            # Calculate final thresholds
            final_correlation = base_thresholds['correlation']
            final_frequency = base_thresholds['frequency']
            
            # Apply adjustments
            final_correlation += size_adjustments[business_size]['correlation']
            final_correlation += industry_adjustments[industry]['correlation']
            final_correlation += risk_adjustments[risk_tolerance]['correlation']
            final_correlation += volume_adjustments[transaction_volume]['correlation']
            final_correlation += seasonality_adjustments[seasonality]['correlation']
            
            final_frequency += size_adjustments[business_size]['frequency']
            final_frequency += industry_adjustments[industry]['frequency']
            final_frequency += risk_adjustments[risk_tolerance]['frequency']
            final_frequency += volume_adjustments[transaction_volume]['frequency']
            final_frequency += seasonality_adjustments[seasonality]['frequency']
            
            # Ensure thresholds stay within valid ranges
            final_correlation = max(0.3, min(0.9, final_correlation))
            final_frequency = max(2, min(10, final_frequency))
            
            # Calculate confidence score for recommendations
            confidence_score = self._calculate_threshold_confidence(
                final_correlation,
                final_frequency,
                business_size,
                industry,
                risk_tolerance
            )
            
            return {
                'correlation_threshold': round(final_correlation, 2),
                'frequency_threshold': int(final_frequency),
                'confidence_score': confidence_score,
                'recommendations': self._generate_threshold_recommendations(
                    final_correlation,
                    final_frequency,
                    confidence_score
                )
            }
        
        except Exception as e:
            print(f"Error in threshold optimization: {str(e)}")
            return None

    def _calculate_threshold_confidence(self,
                                    correlation: float,
                                    frequency: int,
                                    business_size: str,
                                    industry: str,
                                    risk_tolerance: str) -> float:
        """
        Calculate confidence score for threshold recommendations.
        """
        confidence = 1.0
        
        # Adjust confidence based on correlation threshold
        if correlation < 0.5:
            confidence *= 0.8
        elif correlation > 0.8:
            confidence *= 0.9
        
        # Adjust confidence based on frequency threshold
        if frequency < 3:
            confidence *= 0.85
        elif frequency > 7:
            confidence *= 0.9
        
        # Adjust confidence based on business context
        if business_size == 'small' and correlation > 0.7:
            confidence *= 0.9
        if industry == 'services' and frequency > 5:
            confidence *= 0.85
        if risk_tolerance == 'low' and correlation < 0.6:
            confidence *= 0.8
        
        return round(confidence, 2)

    def _generate_threshold_recommendations(self,
                                        correlation: float,
                                        frequency: int,
                                        confidence: float) -> List[str]:
        """
        Generate specific recommendations based on threshold values.
        """
        recommendations = []
        
        if correlation < 0.5:
            recommendations.append(
                "Consider increasing correlation threshold for more reliable bundle identification"
            )
        if correlation > 0.8:
            recommendations.append(
                "High correlation threshold may miss some valid opportunities"
            )
        
        if frequency < 3:
            recommendations.append(
                "Low frequency threshold may lead to less reliable bundle recommendations"
            )
        if frequency > 7:
            recommendations.append(
                "High frequency threshold may be too restrictive for bundle identification"
            )
        
        if confidence < 0.8:
            recommendations.append(
                "Consider adjusting thresholds to improve recommendation confidence"
            )
        
        return recommendations        



def add_cost_optimization_tab(analyzer, df):
    """Add cost optimization analysis tab to the dashboard"""
    st.title("💰 Cost Optimization Analysis")
    
    try:
        # Initialize cost optimization analyzer
        cost_optimizer = CostOptimizationAnalyzer(df)
        
        # Create sub-tabs for different analyses
        bundle_tab, eoq_tab = st.tabs([
            "📦 Bundle Opportunities",
            "📊 Optimal Order Quantities"
        ])
        
        # Bundle Opportunities Analysis
        with bundle_tab:
            st.header("Bundle Purchase Opportunities")
            
            # Add threshold optimization section
            st.subheader("🎯 Threshold Optimization")
            
            # Create columns for business context inputs
            col1, col2 = st.columns(2)
            
            with col1:
                business_size = st.selectbox(
                    "Business Size",
                    options=['small', 'medium', 'large'],
                    index=1,  # Default to 'medium'
                    help="Select your business size to optimize thresholds"
                )
                
                industry = st.selectbox(
                    "Industry Type",
                    options=['manufacturing', 'retail', 'services', 'general'],
                    index=3,  # Default to 'general'
                    help="Select your industry for specialized threshold optimization"
                )
                
                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    options=['low', 'medium', 'high'],
                    index=1,  # Default to 'medium'
                    help="Select your risk tolerance level"
                )
            
            with col2:
                transaction_volume = st.selectbox(
                    "Transaction Volume",
                    options=['low', 'medium', 'high'],
                    index=1,  # Default to 'medium'
                    help="Select your typical transaction volume"
                )
                
                seasonality = st.selectbox(
                    "Seasonality Level",
                    options=['low', 'medium', 'high'],
                    index=0,  # Default to 'low'
                    help="Select your business seasonality level"
                )
            
            # Get optimized thresholds
            optimized_thresholds = cost_optimizer.optimize_bundle_thresholds(
                business_size=business_size,
                industry=industry,
                risk_tolerance=risk_tolerance,
                transaction_volume=transaction_volume,
                seasonality=seasonality
            )
            
            if optimized_thresholds:  # Add null check
                # Display optimization results
                st.subheader("Optimized Thresholds")
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "Correlation Threshold",
                    f"{optimized_thresholds['correlation_threshold']:.2f}",
                    help="Minimum correlation coefficient for bundle identification"
                )
                
                col2.metric(
                    "Frequency Threshold",
                    f"{optimized_thresholds['frequency_threshold']}",
                    help="Minimum number of co-purchases required"
                )
                
                col3.metric(
                    "Confidence Score",
                    f"{optimized_thresholds['confidence_score']:.0%}",
                    help="Confidence level in threshold recommendations"
                )
                
                # Display recommendations
                if optimized_thresholds['recommendations']:
                    st.info("📋 Recommendations:")
                    for rec in optimized_thresholds['recommendations']:
                        st.write(f"• {rec}")
                
                # Add manual override option
                st.subheader("Manual Threshold Adjustment")
                
                manual_override = st.checkbox(
                    "Override optimized thresholds",
                    help="Manually adjust thresholds instead of using optimized values"
                )
                
                if manual_override:
                    col1, col2 = st.columns(2)
                    with col1:
                        min_correlation = st.slider(
                            "Minimum Correlation Threshold",
                            min_value=0.3,
                            max_value=0.9,
                            value=float(optimized_thresholds['correlation_threshold']),
                            step=0.05,
                            help="Minimum correlation coefficient to consider items related"
                        )
                    with col2:
                        min_frequency = st.slider(
                            "Minimum Co-purchase Frequency",
                            min_value=2,
                            max_value=10,
                            value=int(optimized_thresholds['frequency_threshold']),
                            step=1,
                            help="Minimum number of times items should be purchased together"
                        )
                else:
                    min_correlation = optimized_thresholds['correlation_threshold']
                    min_frequency = optimized_thresholds['frequency_threshold']
                
                # Calculate bundle opportunities with optimized/manual thresholds
                bundle_results = cost_optimizer.identify_bundle_opportunities(
                    min_correlation=min_correlation,
                    min_frequency=min_frequency
                )
                
                if bundle_results and bundle_results['bundle_opportunities']:
                    # Display bundle analysis results
                    st.subheader("Bundle Analysis Results")
                    
                    # Summary metrics
                    metrics = bundle_results['metrics']
                    col1, col2, col3 = st.columns(3)
                    
                    col1.metric(
                        "Total Potential Savings",
                        f"${metrics['total_potential_savings']:,.2f}"
                    )
                    col2.metric(
                        "Number of Opportunities",
                        f"{metrics['number_of_opportunities']}"
                    )
                    col3.metric(
                        "Average Correlation",
                        f"{metrics['average_correlation']:.2f}"
                    )
                    
                    # Create visualization of top bundle opportunities
                    if len(bundle_results['bundle_opportunities']) > 0:
                        st.subheader("Top Bundle Opportunities")
                        
                        # Convert to DataFrame for visualization
                        bundle_df = pd.DataFrame(bundle_results['bundle_opportunities'])
                        
                        if not bundle_df.empty:
                            # Create interactive bundle visualization
                            fig = px.bar(
                                bundle_df.head(10),
                                x='potential_savings',
                                y=bundle_df.head(10).apply(
                                    lambda x: f"{x['item1_category']} + {x['item2_category']}", 
                                    axis=1
                                ),
                                orientation='h',
                                title="Top 10 Bundle Opportunities by Potential Savings",
                                labels={
                                    'potential_savings': 'Potential Savings ($)',
                                    'y': 'Bundle Items'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display detailed opportunities table
                            st.subheader("Detailed Bundle Opportunities")
                            
                            # Format DataFrame for display
                            display_df = bundle_df[[
                                'item1_category', 'item2_category',
                                'correlation', 'co_occurrences',
                                'potential_savings'
                            ]].copy()
                            
                            display_df.columns = [
                                'Item 1', 'Item 2', 'Correlation',
                                'Co-purchases', 'Potential Savings'
                            ]
                            
                            st.dataframe(
                                display_df.style.format({
                                    'Correlation': '{:.2f}',
                                    'Potential Savings': '${:,.2f}'
                                }),
                                use_container_width=True
                            )
                            
                            # Add download functionality
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Bundle Opportunities Data",
                                data=csv,
                                file_name="bundle_opportunities.csv",
                                mime="text/csv",
                                key="bundle_opportunities_download"
                            )
                    else:
                        st.warning("No bundle opportunities found with the current thresholds.")
                else:
                    st.warning(
                        "No bundle opportunities found with the current thresholds. "
                        "Try adjusting the thresholds or check your data."
                    )
            else:
                st.error("Error in threshold optimization. Please check your data and try again.")
         # EOQ Analysis Tab
        with eoq_tab:
            st.header("Optimal Order Quantity (EOQ) Analysis")
            
            # Add EOQ parameter inputs
            st.subheader("📈 EOQ Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                holding_cost = st.slider(
                    "Annual Holding Cost (%)",
                    min_value=5,
                    max_value=40,
                    value=20,
                    step=5,
                    help="Annual holding cost as a percentage of unit cost"
                )
                
                carrying_cost_items = st.multiselect(
                    "Included in Holding Cost",
                    options=[
                        "Storage Space",
                        "Insurance",
                        "Material Handling",
                        "Capital Cost",
                        "Obsolescence",
                        "Damage/Spoilage"
                    ],
                    default=[
                        "Storage Space",
                        "Insurance",
                        "Capital Cost"
                    ],
                    help="Components included in holding cost calculation"
                )
            
            with col2:
                ordering_cost = st.number_input(
                    "Ordering Cost ($)",
                    min_value=10,
                    max_value=500,
                    value=50,
                    step=10,
                    help="Fixed cost per order"
                )
                
                ordering_cost_items = st.multiselect(
                    "Included in Ordering Cost",
                    options=[
                        "Purchase Order Processing",
                        "Setup Costs",
                        "Shipping/Freight",
                        "Receiving",
                        "Quality Control",
                        "Administrative"
                    ],
                    default=[
                        "Purchase Order Processing",
                        "Shipping/Freight",
                        "Receiving"
                    ],
                    help="Components included in ordering cost calculation"
                )
            
            # Calculate EOQ
            eoq_results = cost_optimizer.calculate_optimal_order_quantity(
                holding_cost_percentage=holding_cost/100,
                ordering_cost=ordering_cost
            )
            
            if eoq_results and eoq_results['eoq_analysis']:
                # Display summary metrics
                st.subheader("EOQ Analysis Summary")
                
                metrics = eoq_results['metrics']
                col1, col2, col3 = st.columns(3)
                
                col1.metric(
                    "Total Potential Savings",
                    f"${metrics['total_potential_savings']:,.2f}"
                )
                col2.metric(
                    "Average Savings",
                    f"{metrics['average_savings_percentage']:.1f}%"
                )
                col3.metric(
                    "Items Analyzed",
                    f"{metrics['items_analyzed']}"
                )
                
                # Create EOQ visualizations
                st.subheader("EOQ Analysis Insights")
                
                # Convert to DataFrame for visualization
                eoq_df = pd.DataFrame(eoq_results['eoq_analysis'])
                
                # Create comparison visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Current vs Optimal Annual Cost
                    fig = px.bar(
                        eoq_df.head(10),
                        x='category',
                        y=['current_annual_cost', 'optimal_annual_cost'],
                        title="Top 10 Items: Current vs Optimal Annual Cost",
                        barmode='group',
                        labels={
                            'value': 'Annual Cost ($)',
                            'category': 'Category',
                            'variable': 'Cost Type'
                        }
                    )
                    fig.update_layout(legend_title_text='Cost Type')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Order Size Comparison
                    fig = px.bar(
                        eoq_df.head(10),
                        x='category',
                        y=['current_order_size', 'eoq'],
                        title="Current vs Optimal Order Sizes",
                        barmode='group',
                        labels={
                            'value': 'Order Size',
                            'category': 'Category',
                            'variable': 'Order Size Type'
                        }
                    )
                    fig.update_layout(legend_title_text='Order Size Type')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Cost Breakdown Analysis
                st.subheader("Cost Breakdown Analysis")
                
                selected_category = st.selectbox(
                    "Select Category for Detailed Analysis",
                    options=eoq_df['category'].unique(),
                    index=0
                )
                
                # Filter data for selected category
                category_data = eoq_df[eoq_df['category'] == selected_category].iloc[0]
                
                # Create cost breakdown visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Current Cost Structure
                    current_costs = {
                        'Ordering Cost': category_data['current_annual_cost'] * 0.4,  # Example split
                        'Holding Cost': category_data['current_annual_cost'] * 0.6
                    }
                    
                    fig = px.pie(
                        values=list(current_costs.values()),
                        names=list(current_costs.keys()),
                        title="Current Cost Structure"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Optimal Cost Structure
                    optimal_costs = {
                        'Ordering Cost': category_data['optimal_annual_cost'] * 0.5,  # EOQ balances these costs
                        'Holding Cost': category_data['optimal_annual_cost'] * 0.5
                    }
                    
                    fig = px.pie(
                        values=list(optimal_costs.values()),
                        names=list(optimal_costs.keys()),
                        title="Optimal Cost Structure"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed EOQ Results Table
                st.subheader("Detailed EOQ Analysis")
                
                # Format DataFrame for display
                display_df = eoq_df[[
                    'category', 'supplier', 'annual_demand', 'unit_cost',
                    'eoq', 'current_order_size', 'optimal_orders_per_year',
                    'current_annual_cost', 'optimal_annual_cost', 'potential_savings',
                    'savings_percentage'
                ]].copy()
                
                display_df.columns = [
                    'Category', 'Supplier', 'Annual Demand', 'Unit Cost',
                    'Optimal Order Quantity', 'Current Order Size', 
                    'Optimal Orders per Year', 'Current Annual Cost',
                    'Optimal Annual Cost', 'Potential Savings', 'Savings %'
                ]
                
                st.dataframe(
                    display_df.style.format({
                        'Unit Cost': '${:,.2f}',
                        'Current Annual Cost': '${:,.2f}',
                        'Optimal Annual Cost': '${:,.2f}',
                        'Potential Savings': '${:,.2f}',
                        'Savings %': '{:.1f}%',
                        'Annual Demand': '{:,.0f}',
                        'Optimal Order Quantity': '{:,.0f}',
                        'Current Order Size': '{:,.0f}',
                        'Optimal Orders per Year': '{:.1f}'
                    }),
                    use_container_width=True
                )
                
                # Download button for EOQ analysis
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download EOQ Analysis Data",
                    data=csv,
                    file_name="eoq_analysis.csv",
                    mime="text/csv",
                     key="eoq_analysis_download"
                )
                
                # EOQ Implementation Recommendations
                st.subheader("Implementation Recommendations")
                
                # Generate recommendations based on analysis
                recommendations = []
                
                # High savings opportunities
                high_savings = eoq_df[eoq_df['savings_percentage'] > 20]
                if not high_savings.empty:
                    recommendations.append(
                        f"Focus on {len(high_savings)} items with >20% potential savings"
                    )
                
                # Large order size adjustments
                size_diff = abs(eoq_df['current_order_size'] - eoq_df['eoq']) / eoq_df['current_order_size']
                large_adjustments = eoq_df[size_diff > 0.5]
                if not large_adjustments.empty:
                    recommendations.append(
                        f"Significant order size adjustments needed for {len(large_adjustments)} items"
                    )
                
                # Display recommendations
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
                
                # Implementation timeline
                st.write("Suggested Implementation Timeline:")
                timeline_data = {
                    'Phase': ['Immediate', 'Short-term', 'Medium-term'],
                    'Actions': [
                        'Adjust order sizes for high-savings items',
                        'Update ordering processes and systems',
                        'Monitor and optimize results'
                    ],
                    'Timeline': ['1-2 weeks', '1-2 months', '3-6 months']
                }
                
                timeline_df = pd.DataFrame(timeline_data)
                st.table(timeline_df)
            
            else:
                st.warning("Unable to perform EOQ analysis. Please check your data.")       
    except Exception as e:
        st.error(f"Error in bundle analysis: {str(e)}")
        st.info("Please ensure your data contains all required fields and is properly formatted.")

def prepare_consolidated_excel(analyzer, df):
    """
    Prepare a consolidated Excel file containing all analyses using openpyxl
    """
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. Executive Summary
            summary_data = {
                'Metric': [
                    'Report Generated',
                    'Analysis Period',
                    'Total Spend',
                    'Total Savings',
                    'Number of Suppliers',
                    'Number of Categories',
                    'Number of Projects'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{df['Transaction Date'].min()} to {df['Transaction Date'].max()}",
                    f"${df['Total Value'].sum():,.2f}",
                    f"${df['CostSavings'].sum():,.2f}",
                    len(df['Supplier'].unique()),
                    len(df['Category'].unique()),
                    len(df['Project'].unique())
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive Summary', index=False)

            # 2. Spend Analysis
            spend_results = analyzer.analyze_spend()
            if spend_results:
                monthly_spend = pd.DataFrame({
                    'Month': list(spend_results['monthly_spend'].keys()),
                    'Spend': list(spend_results['monthly_spend'].values())
                })
                monthly_spend.to_excel(writer, sheet_name='Spend Analysis', index=False)

                category_spend = pd.DataFrame({
                    'Category': list(spend_results['category_spend'].keys()),
                    'Spend': list(spend_results['category_spend'].values())
                }).sort_values('Spend', ascending=False)
                category_spend.to_excel(writer, sheet_name='Category Spend', index=False)

            # 3. Supplier Performance
            supplier_results = analyzer.analyze_supplier_performance()
            if supplier_results:
                supplier_data = {
                    'Supplier': list(supplier_results['avg_lead_time'].keys()),
                    'Avg Lead Time (Days)': list(supplier_results['avg_lead_time'].values()),
                    'On-Time Delivery Rate (%)': list(supplier_results['delivery_performance']['on_time_rate'].values()),
                    'Total Deliveries': list(supplier_results['delivery_performance']['total_deliveries'].values()),
                    'Average Unit Price ($)': list(supplier_results['price_stats']['avg_price'].values()),
                    'Total Savings ($)': list(supplier_results['price_stats']['total_savings'].values())
                }
                pd.DataFrame(supplier_data).to_excel(writer, sheet_name='Supplier Performance', index=False)

            # 4. Category Analysis
            category_results = analyzer.analyze_category_insights()
            if category_results:
                category_data = pd.DataFrame({
                    'Category': list(category_results['category_volume'].keys()),
                    'Volume': list(category_results['category_volume'].values()),
                    'Savings': list(category_results['category_savings'].values())
                })
                category_data.to_excel(writer, sheet_name='Category Analysis', index=False)

            # 5. Project Performance
            project_results = analyzer.analyze_project_performance()
            if project_results:
                projects = list(project_results['project_spend'].keys())
                project_data = pd.DataFrame({
                    'Project': projects,
                    'Total Spend ($)': [project_results['project_spend'][p] for p in projects],
                    'Total Deliveries': [project_results['delivery_performance']['total_deliveries'][p] for p in projects],
                    'On-Time Rate (%)': [project_results['delivery_performance']['on_time_rate'][p] for p in projects],
                    'Material Quantity': [project_results['material_analysis']['quantities'][p] for p in projects]
                })
                project_data.to_excel(writer, sheet_name='Project Performance', index=False)

            # 6. Cost Optimization
            cost_optimizer = CostOptimizationAnalyzer(df)
            bundle_results = cost_optimizer.identify_bundle_opportunities()
            eoq_results = cost_optimizer.calculate_optimal_order_quantity()

            if bundle_results and bundle_results['bundle_opportunities']:
                pd.DataFrame(bundle_results['bundle_opportunities']).to_excel(
                    writer, sheet_name='Bundle Opportunities', index=False
                )

            if eoq_results and eoq_results['eoq_analysis']:
                pd.DataFrame(eoq_results['eoq_analysis']).to_excel(
                    writer, sheet_name='EOQ Analysis', index=False
                )

    except ImportError as e:
        st.error("Excel writer not available. Installing required dependencies...")
        try:
            import subprocess
            subprocess.check_call(["pip", "install", "openpyxl"])
            st.success("Dependencies installed successfully. Please try again.")
            return None
        except Exception as install_error:
            st.error(f"Error installing dependencies: {str(install_error)}")
            return None
    except Exception as e:
        st.error(f"Error generating Excel file: {str(e)}")
        return None

    return output.getvalue()


def main():
    st.title("📊 Procurement Analysis Dashboard")
    try:
        uploaded_file = st.sidebar.file_uploader("Upload Procurement Data", type="csv")
        
        if uploaded_file is not None:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_columns = [
                'Transaction Date', 'Supplier', 'Category', 
                'Project', 'Quantity', 'SupplierUnit Price', 
                'CostSavings', 'Delivery Status'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return
            
            analyzer = ProcurementAnalyzer(df)
            
            # Debug information toggle
            if st.checkbox("Show Data Info"):
                st.write("DataFrame Info:")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
                st.write("Data Sample:")
                st.write(df.head())
                
                st.write("Missing Values:")
                st.write(df.isnull().sum())

            # Add consolidated report generation to sidebar
            st.sidebar.markdown("---")
            st.sidebar.header("📥 Download Complete Analysis")
            
            if st.sidebar.button("Generate Complete Report", key="generate_complete_report"):
                try:
                    with st.spinner('Generating comprehensive report...'):
                        excel_data = prepare_consolidated_excel(analyzer, df)
                        if excel_data is not None:
                            st.sidebar.download_button(
                                label="📥 Download Complete Analysis (Excel)",
                                data=excel_data,
                                file_name=f"procurement_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="download_complete_report"
                            )
                            st.sidebar.success("Report generated successfully! Click above to download.")
                except Exception as e:
                    st.sidebar.error(f"Error generating report: {str(e)}")


            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4,tab5 = st.tabs([
                "💰 Spend Analysis",
                "🏢 Supplier Performance",
                "📦 Category Insights",
                "📋 Project Analysis",
                "💡 Cost Optimization"
            ])
            
# Tab 1: Spend Analysis
            with tab1:
                spend_results = analyzer.analyze_spend()
                if spend_results:
                    st.header("Spend Analysis")
                    
                    # Create all DataFrames first
                    monthly_df = pd.DataFrame({
                        'Month': list(spend_results['monthly_spend'].keys()),
                        'Spend': list(spend_results['monthly_spend'].values())
                    })
                    
                    yearly_df = pd.DataFrame({
                        'Year': list(spend_results['yearly_spend'].keys()),
                        'Spend': list(spend_results['yearly_spend'].values())
                    })
                    
                    supplier_df = pd.DataFrame({
                        'Supplier': list(spend_results['supplier_spend'].keys()),
                        'Spend': list(spend_results['supplier_spend'].values())
                    }).sort_values('Spend', ascending=False)
                    
                    category_df = pd.DataFrame({
                        'Category': list(spend_results['category_spend'].keys()),
                        'Spend': list(spend_results['category_spend'].values())
                    }).sort_values('Spend', ascending=False)
                    
                    project_spend_df = pd.DataFrame({
                        'Project': list(spend_results['project_spend'].keys()),
                        'Spend': list(spend_results['project_spend'].values())
                    }).sort_values('Spend', ascending=False)
                    
                    project_savings_df = pd.DataFrame({
                        'Project': list(spend_results['project_savings'].keys()),
                        'Savings': list(spend_results['project_savings'].values())
                    }).sort_values('Savings', ascending=False)
                    
                    # Calculate summary metrics
                    total_spend = sum(spend_results['monthly_spend'].values())
                    total_savings = sum(spend_results['project_savings'].values())
                    savings_rate = (total_savings / total_spend * 100) if total_spend > 0 else 0
                    
                    summary_df = pd.DataFrame({
                        'Metric': ['Total Spend', 'Total Savings', 'Savings Rate'],
                        'Value': [f"${total_spend:,.2f}", f"${total_savings:,.2f}", f"{savings_rate:.1f}%"]
                    })
                    
                    # Combined project data
                    project_combined_df = project_spend_df.merge(
                        project_savings_df,
                        on='Project',
                        how='outer'
                    ).fillna(0)
                    
                    # Monthly and Yearly Spend Visualizations
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Monthly Spend Trend")
                        fig = px.line(
                            monthly_df,
                            x='Month',
                            y='Spend',
                            title="Monthly Spend Trend",
                            labels={'Spend': 'Total Spend ($)', 'Month': 'Month'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Yearly Spend Pattern")
                        fig = px.bar(
                            yearly_df,
                            x='Year',
                            y='Spend',
                            title="Yearly Spend Pattern",
                            labels={'Spend': 'Total Spend ($)', 'Year': 'Year'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Spend Distribution
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Spend by Supplier")
                        fig = px.pie(
                            supplier_df,
                            values='Spend',
                            names='Supplier',
                            title="Supplier Spend Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        st.subheader("Spend by Category")
                        fig = px.pie(
                            category_df,
                            values='Spend',
                            names='Category',
                            title="Category Spend Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Project Analysis
                    st.subheader("Project Analysis")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            project_spend_df,
                            x='Project',
                            y='Spend',
                            title="Project Spend Distribution",
                            labels={'Spend': 'Total Spend ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            project_savings_df,
                            x='Project',
                            y='Savings',
                            title="Project Cost Savings",
                            labels={'Savings': 'Total Savings ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary metrics display
                    st.subheader("Summary Metrics")
                    metric1, metric2, metric3 = st.columns(3)
                    metric1.metric("Total Spend", f"${total_spend:,.2f}")
                    metric2.metric("Total Savings", f"${total_savings:,.2f}")
                    metric3.metric("Savings Rate", f"{savings_rate:.1f}%")

                    # Complete spend analysis download
                    # Create Excel file with all analysis data
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        # Write each DataFrame to a separate sheet
                        summary_df.to_excel(writer, sheet_name='Summary Metrics', index=False)
                        monthly_df.to_excel(writer, sheet_name='Monthly Spend', index=False)
                        yearly_df.to_excel(writer, sheet_name='Yearly Spend', index=False)
                        supplier_df.to_excel(writer, sheet_name='Supplier Spend', index=False)
                        category_df.to_excel(writer, sheet_name='Category Spend', index=False)
                        project_combined_df.to_excel(writer, sheet_name='Project Analysis', index=False)
                    
                    # Download complete analysis button
                    st.download_button(
                        label="📥 Download Complete Spend Analysis",
                        data=output.getvalue(),
                        file_name="complete_spend_analysis.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="spend_analysis_download"
                    )


            # Tab 2: Supplier Performance
            with tab2:
                supplier_results = analyzer.analyze_supplier_performance()
                if supplier_results:
                    st.header("Supplier Performance Analysis")
                    
                    # Summary Metrics
                    st.subheader("Performance Summary")
                    total_suppliers = len(supplier_results['avg_lead_time'])
                    avg_otd = np.mean(list(supplier_results['delivery_performance']['on_time_rate'].values()))
                    total_savings = sum(supplier_results['price_stats']['total_savings'].values())
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Suppliers", f"{total_suppliers}")
                    col2.metric("Average On-Time Delivery", f"{avg_otd:.1f}%")
                    col3.metric("Total Cost Savings", f"${total_savings:,.2f}")
                    
                    # Lead Time Analysis
                    st.subheader("Lead Time Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        lead_time_df = pd.DataFrame({
                            'Supplier': list(supplier_results['avg_lead_time'].keys()),
                            'Lead Time': list(supplier_results['avg_lead_time'].values())
                        }).sort_values('Lead Time', ascending=True)
                        
                        fig = px.bar(
                            lead_time_df,
                            x='Supplier',
                            y='Lead Time',
                            title="Average Lead Time by Supplier",
                            labels={'Lead Time': 'Days'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        delivery_df = pd.DataFrame({
                            'Supplier': list(supplier_results['delivery_performance']['on_time_rate'].keys()),
                            'On-Time Rate': list(supplier_results['delivery_performance']['on_time_rate'].values())
                        }).sort_values('On-Time Rate', ascending=False)
                        
                        fig = px.bar(
                            delivery_df,
                            x='Supplier',
                            y='On-Time Rate',
                            title="On-Time Delivery Performance",
                            labels={'On-Time Rate': 'On-Time Delivery Rate (%)'}
                        )
                        fig.add_hline(y=95, line_dash="dash", line_color="red",
                                    annotation_text="Target (95%)")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Price Analysis
                    st.subheader("Price Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = go.Figure()
                        for supplier, prices in supplier_results['price_distribution'].items():
                            if prices:
                                fig.add_trace(go.Box(
                                    y=prices,
                                    name=supplier,
                                    boxpoints='outliers'
                                ))
                        fig.update_layout(
                            title="Price Distribution by Supplier",
                            yaxis_title="Unit Price ($)",
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        savings_df = pd.DataFrame({
                            'Supplier': list(supplier_results['price_stats']['total_savings'].keys()),
                            'Savings': list(supplier_results['price_stats']['total_savings'].values())
                        }).sort_values('Savings', ascending=True)
                        
                        fig = px.bar(
                            savings_df,
                            x='Supplier',
                            y='Savings',
                            title="Total Cost Savings by Supplier",
                            labels={'Savings': 'Total Savings ($)'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed Performance Table
                    st.subheader("Detailed Supplier Performance")
                    
                    performance_data = {
                        'Supplier': list(supplier_results['avg_lead_time'].keys()),
                        'Avg Lead Time (Days)': list(supplier_results['avg_lead_time'].values()),
                        'On-Time Delivery Rate (%)': list(supplier_results['delivery_performance']['on_time_rate'].values()),
                        'Total Deliveries': list(supplier_results['delivery_performance']['total_deliveries'].values()),
                        'Average Unit Price ($)': list(supplier_results['price_stats']['avg_price'].values()),
                        'Total Savings ($)': list(supplier_results['price_stats']['total_savings'].values())
                    }
                    
                    performance_df = pd.DataFrame(performance_data)
                    formatted_df = performance_df.copy()
                    formatted_df['Avg Lead Time (Days)'] = formatted_df['Avg Lead Time (Days)'].round(1)
                    formatted_df['On-Time Delivery Rate (%)'] = formatted_df['On-Time Delivery Rate (%)'].round(1)
                    formatted_df['Average Unit Price ($)'] = formatted_df['Average Unit Price ($)'].round(2)
                    formatted_df['Total Savings ($)'] = formatted_df['Total Savings ($)'].round(2)
                    
                    st.dataframe(
                        formatted_df.style.format({
                            'Total Savings ($)': '${:,.2f}',
                            'Average Unit Price ($)': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = formatted_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Supplier Performance Data",
                        data=csv,
                        file_name="supplier_performance.csv",
                        mime="text/csv",
                        key="supplier_performance_download"
                    )
            # Tab 3: Category Insights
            with tab3:
                category_results = analyzer.analyze_category_insights()
                if category_results:
                    try:
                        st.header("Category and Commodity Analysis")
                        
                        # Summary Metrics
                        total_categories = len(category_results['category_volume'])
                        total_commodities = len(category_results['commodity_spend'])
                        total_category_savings = sum(category_results['category_savings'].values())
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Categories", str(total_categories))
                        col2.metric("Total Commodities", str(total_commodities))
                        col3.metric("Total Category Savings", f"${total_category_savings:,.2f}")
                        
                        # Category Analysis
                        st.subheader("Category Analysis")
                        
                        # Create DataFrames for visualization
                        category_volume_df = pd.DataFrame({
                            'Category': list(category_results['category_volume'].keys()),
                            'Volume': list(category_results['category_volume'].values())
                        }).sort_values('Volume', ascending=True)
                        
                        category_savings_df = pd.DataFrame({
                            'Category': list(category_results['category_savings'].keys()),
                            'Savings': list(category_results['category_savings'].values())
                        }).sort_values('Savings', ascending=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.bar(
                                category_volume_df,
                                x='Category',
                                y='Volume',
                                title="Volume by Category",
                                labels={'Volume': 'Total Volume'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.bar(
                                category_savings_df,
                                x='Category',
                                y='Savings',
                                title="Savings by Category",
                                labels={'Savings': 'Total Savings ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Price Trends
                        st.subheader("Price Trends Analysis")
                        
                        # Convert price trends to DataFrame
                        price_trends_data = pd.DataFrame([
                            {
                                'Month': key.split('_')[0],
                                'Category': key.split('_')[1],
                                'Price': value
                            }
                            for key, value in category_results['price_trends'].items()
                        ])
                        
                        if not price_trends_data.empty:
                            fig = px.line(
                                price_trends_data,
                                x='Month',
                                y='Price',
                                color='Category',
                                title="Price Trends by Category",
                                labels={'Price': 'Average Unit Price ($)', 'Month': 'Month'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Category Performance Summary
                        st.subheader("Category Performance Summary")
                        
                        summary_data = []
                        for category in category_results['category_volume'].keys():
                            summary_data.append({
                                'Category': category,
                                'Total Volume': category_results['category_volume'][category],
                                'Total Savings ($)': category_results['category_savings'][category]
                            })
                        
                        category_summary = pd.DataFrame(summary_data)
                        
                        # Display formatted table
                        st.dataframe(
                            category_summary.style.format({
                                'Total Savings ($)': '${:,.2f}',
                                'Total Volume': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = category_summary.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Category Performance Data",
                            data=csv,
                            file_name="category_performance.csv",
                            mime="text/csv",
                            key="category_performance_download"
                        )
                    
                    except Exception as e:
                        st.error(f"Error in Category Analysis Visualization: {str(e)}")
                        print(f"Detailed visualization error: {e}")  # For debugging
                        st.info("Please ensure all required category data is available and properly formatted.")


            # Tab 4: Project Analysis
            with tab4:
                project_results = analyzer.analyze_project_performance()
                if project_results:
                    try:
                        st.header("Project Performance Analysis")
                        
                        # Get list of projects
                        projects = list(project_results['project_spend'].keys())
                        
                        # Create DataFrame for visualization
                        performance_data = pd.DataFrame({
                            'Project': projects,
                            'Total Spend ($)': [project_results['project_spend'][p] for p in projects],
                            'Total Deliveries': [project_results['delivery_performance']['total_deliveries'][p] for p in projects],
                            'On-Time Rate (%)': [project_results['delivery_performance']['on_time_rate'][p] for p in projects],
                            'Total Quantity': [project_results['material_analysis']['quantities'][p] for p in projects]
                        })
                        
                        # Summary Metrics
                        st.subheader("Project Overview")
                        total_spend = performance_data['Total Spend ($)'].sum()
                        avg_otd = performance_data['On-Time Rate (%)'].mean()
                        total_deliveries = performance_data['Total Deliveries'].sum()
                        total_quantity = performance_data['Total Quantity'].sum()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Project Spend", f"${total_spend:,.2f}")
                        col2.metric("Average On-Time Delivery", f"{avg_otd:.1f}%")
                        col3.metric("Total Deliveries", f"{total_deliveries:,}")
                        col4.metric("Total Quantity", f"{total_quantity:,}")
                        
                        # Project Spend Analysis
                        st.subheader("Project Spend Analysis")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            spend_df = performance_data.sort_values('Total Spend ($)', ascending=True)
                            fig = px.bar(
                                spend_df,
                                x='Project',
                                y='Total Spend ($)',
                                title="Total Spend by Project",
                                labels={'Total Spend ($)': 'Total Spend ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            delivery_df = performance_data.sort_values('On-Time Rate (%)', ascending=False)
                            fig = px.bar(
                                delivery_df,
                                x='Project',
                                y='On-Time Rate (%)',
                                title="Project Delivery Performance",
                                labels={'On-Time Rate (%)': 'On-Time Delivery Rate (%)'}
                            )
                            fig.add_hline(
                                y=95, 
                                line_dash="dash", 
                                line_color="red",
                                annotation_text="Target (95%)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Material Analysis
                        if project_results['material_analysis']['category_breakdown']:
                            st.subheader("Material Analysis")
                            
                            # Convert material analysis data to DataFrame
                            material_df = pd.DataFrame(project_results['material_analysis']['category_breakdown'])
                            
                            # Create stacked bar chart for material distribution
                            fig = px.bar(
                                material_df,
                                x='Project',
                                y='Total Value',
                                color='Category',
                                title="Material Cost Distribution by Project and Category",
                                labels={'Total Value': 'Total Cost ($)'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Material Distribution Pie Charts
                            st.subheader("Project Material Distribution")
                            for project in projects:
                                project_data = material_df[material_df['Project'] == project]
                                if not project_data.empty:
                                    fig = px.pie(
                                        project_data,
                                        values='Total Value',
                                        names='Category',
                                        title=f"{project} Material Distribution"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed Project Performance Table
                        st.subheader("Detailed Project Performance")
                        
                        # Calculate additional metrics
                        performance_df = performance_data.copy()
                        performance_df['Average Cost per Delivery ($)'] = (
                            performance_df['Total Spend ($)'] / performance_df['Total Deliveries'].replace(0, 1)
                        ).round(2)
                        
                        performance_df['Average Quantity per Delivery'] = (
                            performance_df['Total Quantity'] / performance_df['Total Deliveries'].replace(0, 1)
                        ).round(2)
                        
                        # Format and display table
                        st.dataframe(
                            performance_df.style.format({
                                'Total Spend ($)': '${:,.2f}',
                                'On-Time Rate (%)': '{:.1f}%',
                                'Average Cost per Delivery ($)': '${:,.2f}'
                            }),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = performance_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Project Performance Data",
                            data=csv,
                            file_name="project_performance.csv",
                            mime="text/csv",
                            key="project_performance_download"
                        )
                    
                    except Exception as e:
                        st.error(f"Error in Project Analysis Visualization: {str(e)}")
                        print(f"Detailed visualization error: {e}")  # For debugging
                        st.info("Please ensure all required project data is available and properly formatted.")

                # New Cost Optimization tab
            with tab5:
                add_cost_optimization_tab(analyzer, df)
                

        
        else:
            # Welcome message when no file is uploaded
            st.markdown("""
                ### 👋 Welcome to the Procurement Analysis Dashboard
                
                This dashboard provides comprehensive analysis of procurement data including:
                
                #### 📊 Key Features:
                - Spend Analysis
                - Supplier Performance Metrics
                - Category Insights
                - Project Performance Analysis
                - Cost Optimization        
                
                #### 📁 Required Data Format:
                Please upload a CSV file containing the following columns:
                - Transaction Date
                - Supplier
                - Category
                - Commodity
                - Project
                - Quantity
                - SupplierUnit Price
                - Lead Time (Days)
                - Delivery Status
                - CostSavings
                
                #### 🎯 Get Started:
                Upload your procurement data using the file uploader in the sidebar to begin the analysis.
            """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your data format and try again.")

if __name__ == "__main__":
    main()