"""
Backend module for Publication Analytics Dashboard
Handles data loading, processing, and statistical computations
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PublicationDataProcessor:
    """Handles all data processing and analytics operations"""
    
    def __init__(self, filepath: str = 'publications.csv'):
        self.filepath = filepath
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and clean the publications dataset"""
        self.df = pd.read_csv(self.filepath)
        self.df.columns = self.df.columns.str.strip()
        return self.df
    
    def filter_data(self, 
                   year_range: Tuple[int, int],
                   countries: List[str],
                   max_rank: int) -> pd.DataFrame:
        """Apply filters to the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        filtered = self.df[
            (self.df['year'] >= year_range[0]) & 
            (self.df['year'] <= year_range[1]) &
            (self.df['Rank'] <= max_rank)
        ]
        
        if 'All' not in countries and len(countries) > 0:
            filtered = filtered[filtered['Name'].isin(countries)]
        
        return filtered
    
    def get_summary_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key summary metrics"""
        return {
            'total_publications': df['Web of Science Documents'].sum(),
            'total_citations': df['Times Cited'].sum(),
            'unique_countries': len(df['Name'].unique()),
            'avg_collaboration': df['Collab-CNCI'].mean()
        }
    
    def get_top_countries(self, 
                         df: pd.DataFrame, 
                         metric: str, 
                         n: int = 10) -> pd.Series:
        """Get top N countries by specified metric"""
        return df.groupby('Name')[metric].sum().sort_values(ascending=False).head(n)
    
    def calculate_citation_efficiency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate citations per document"""
        result = df.copy()
        result['Citation_Per_Doc'] = result['Times Cited'] / result['Web of Science Documents']
        return result.nlargest(15, 'Citation_Per_Doc')[['Name', 'Citation_Per_Doc', 'year']]
    
    def get_country_data(self, country: str) -> pd.DataFrame:
        """Get all data for a specific country"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df[self.df['Name'] == country]
    
    def get_country_metrics(self, country: str) -> Dict[str, float]:
        """Get summary metrics for a specific country"""
        country_data = self.get_country_data(country)
        return {
            'total_publications': country_data['Web of Science Documents'].sum(),
            'total_citations': country_data['Times Cited'].sum(),
            'avg_rank': country_data['Rank'].mean(),
            'avg_collaboration': country_data['Collab-CNCI'].mean()
        }
    
    def compare_country_to_global(self, 
                                  country: str, 
                                  metrics: List[str]) -> pd.DataFrame:
        """Compare country metrics to global averages"""
        country_data = self.get_country_data(country)
        country_means = country_data[metrics].mean()
        global_means = self.df[metrics].mean()
        
        return pd.DataFrame({
            'Metric': metrics,
            country: country_means.values,
            'Global Average': global_means.values
        })
    
    def get_temporal_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by year"""
        yearly = df.groupby('year').agg({
            'Web of Science Documents': 'sum',
            'Times Cited': 'sum',
            'Collab-CNCI': 'mean',
            'Name': 'count'
        }).reset_index()
        yearly.columns = ['Year', 'Publications', 'Citations', 
                         'Avg_Collaboration', 'Country_Count']
        return yearly
    
    def calculate_growth_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year growth rates"""
        result = df.copy()
        result['Pub_Growth'] = result['Publications'].pct_change() * 100
        result['Citation_Growth'] = result['Citations'].pct_change() * 100
        return result
    
    def create_publication_heatmap_data(self, 
                                       df: pd.DataFrame, 
                                       top_n: int = 20) -> pd.DataFrame:
        """Create pivot table for publication heatmap"""
        pivot = df.pivot_table(
            values='Web of Science Documents',
            index='Name',
            columns='year',
            aggfunc='sum',
            fill_value=0
        )
        
        top_countries = df.groupby('Name')['Web of Science Documents'].sum().nlargest(top_n).index
        return pivot.loc[top_countries]
    
    def calculate_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for numeric features"""
        numeric_cols = [
            'Web of Science Documents', 'Times Cited', 'Collab-CNCI', 
            '% Docs Cited', 'Category Normalized Citation Impact', 'Rank'
        ]
        return df[numeric_cols].corr()
    
    def detect_outliers(self, 
                       df: pd.DataFrame, 
                       metric: str = 'Times Cited', 
                       threshold: float = 2.0) -> pd.DataFrame:
        """Detect statistical outliers using Z-score"""
        z_scores = np.abs(stats.zscore(df[metric]))
        outlier_df = df.copy()
        outlier_df['Z_Score'] = z_scores
        return outlier_df[z_scores > threshold][
            ['Name', metric, 'year', 'Z_Score']
        ].sort_values('Z_Score', ascending=False)
    
    def create_performance_quadrants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize entries into performance quadrants"""
        result = df.copy()
        median_pubs = df['Web of Science Documents'].median()
        median_citations = df['Times Cited'].median()
        
        def assign_quadrant(row):
            if row['Web of Science Documents'] > median_pubs and row['Times Cited'] > median_citations:
                return 'High Pub, High Cit'
            elif row['Web of Science Documents'] > median_pubs:
                return 'High Pub, Low Cit'
            elif row['Times Cited'] > median_citations:
                return 'Low Pub, High Cit'
            else:
                return 'Low Pub, Low Cit'
        
        result['Quadrant'] = result.apply(assign_quadrant, axis=1)
        result['Median_Pubs'] = median_pubs
        result['Median_Citations'] = median_citations
        return result
    
    def get_ranking_dynamics(self, 
                            df: pd.DataFrame, 
                            top_n: int = 15) -> pd.DataFrame:
        """Analyze ranking dynamics by country"""
        rank_stats = df.groupby('Name')['Rank'].agg([
            'mean', 'min', 'max', 'std'
        ]).sort_values('mean')
        return rank_stats.head(top_n)
    
    def get_statistical_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get statistical summary of key metrics"""
        key_metrics = [
            'Web of Science Documents', 'Times Cited', 
            'Collab-CNCI', '% Docs Cited'
        ]
        return df[key_metrics].describe()
    
    def calculate_collaboration_correlation(self, df: pd.DataFrame) -> float:
        """Calculate correlation between collaboration index and citations"""
        return df['Collab-CNCI'].corr(df['Times Cited'])
    
    def get_available_countries(self) -> List[str]:
        """Get list of all countries in dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return sorted(self.df['Name'].unique().tolist())
    
    def get_year_range(self) -> Tuple[int, int]:
        """Get min and max years in dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return int(self.df['year'].min()), int(self.df['year'].max())
    
    def get_max_rank(self) -> int:
        """Get maximum rank value in dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return int(self.df['Rank'].max())
    
    def export_to_csv(self, df: pd.DataFrame) -> str:
        """Export dataframe to CSV string"""
        return df.to_csv(index=False)


# Utility functions
def format_number(num: float, decimal_places: int = 0) -> str:
    """Format number with thousands separator"""
    if decimal_places == 0:
        return f"{num:,.0f}"
    return f"{num:,.{decimal_places}f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values"""
    if old_value == 0:
        return 0
    return ((new_value - old_value) / old_value) * 100