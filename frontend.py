"""
Frontend module for Publication Analytics Dashboard
Handles UI components and visualizations using Streamlit
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend import PublicationDataProcessor, format_number
import os


# Page configuration
st.set_page_config(
    page_title="Publication Analytics Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)


def load_css(css_file):
    """
    Load and inject external CSS file into Streamlit app
    
    Args:
        css_file (str): Path to the CSS file
    """
    try:
        if not os.path.exists(css_file):
            st.warning(f"CSS file '{css_file}' not found. Using default styling.")
            return
        
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
            st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"❌ Error loading CSS file: {str(e)}")


# Load custom styles from external CSS file
load_css('styles.css')


@st.cache_resource
def get_data_processor():
    """Initialize and cache the data processor"""
    processor = PublicationDataProcessor('publications.csv')
    processor.load_data()
    return processor


def render_overview(processor, filtered_df):
    """Render the Overview section"""
    st.header("Executive Summary")
    
    # Key metrics
    metrics = processor.get_summary_metrics(filtered_df)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Publications", format_number(metrics['total_publications']))
    with col2:
        st.metric("Total Citations", format_number(metrics['total_citations']))
    with col3:
        st.metric("Countries Analyzed", int(metrics['unique_countries']))
    with col4:
        st.metric("Avg Collaboration Index", f"{metrics['avg_collaboration']:.2f}")
    
    st.markdown("---")
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top 10 Countries by Citations")
        top_citations = processor.get_top_countries(filtered_df, 'Times Cited', 10)
        fig = px.bar(
            x=top_citations.values,
            y=top_citations.index,
            orientation='h',
            labels={'x': 'Total Citations', 'y': 'Country'},
            color=top_citations.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top 10 Countries by Publications")
        top_pubs = processor.get_top_countries(filtered_df, 'Web of Science Documents', 10)
        fig = px.bar(
            x=top_pubs.values,
            y=top_pubs.index,
            orientation='h',
            labels={'x': 'Total Publications', 'y': 'Country'},
            color=top_pubs.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Citation efficiency
    st.subheader("⚡ Citation Efficiency Analysis")
    top_efficiency = processor.calculate_citation_efficiency(filtered_df)
    
    fig = px.scatter(
        top_efficiency,
        x='year',
        y='Citation_Per_Doc',
        size='Citation_Per_Doc',
        color='Name',
        hover_data=['Name'],
        title="Top 15 Countries by Citation Efficiency (Citations per Document)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights
    st.markdown("### 💡 Key Insights")
    st.markdown("""
    <div class="insight-box">
    <b>Finding 1:</b> Countries with higher collaboration indices tend to have better citation rates, 
    indicating the importance of international partnerships.
    </div>
    """, unsafe_allow_html=True)


def render_country_analysis(processor):
    """Render the Country Analysis section"""
    st.header("Country Deep Dive")
    
    countries = processor.get_available_countries()
    country = st.selectbox("Select a Country for Detailed Analysis", countries)
    
    country_data = processor.get_country_data(country)
    metrics = processor.get_country_metrics(country)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Publications", format_number(metrics['total_publications']))
    with col2:
        st.metric("Total Citations", format_number(metrics['total_citations']))
    with col3:
        st.metric("Avg Rank", f"{metrics['avg_rank']:.1f}")
    
    # Temporal evolution
    st.subheader(f"{country}'s Research Evolution Over Time")
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Publications Trend', 'Citations Trend'),
        vertical_spacing=0.15
    )
    
    fig.add_trace(
        go.Scatter(
            x=country_data['year'], 
            y=country_data['Web of Science Documents'],
            mode='lines+markers', 
            name='Publications', 
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=country_data['year'], 
            y=country_data['Times Cited'],
            mode='lines+markers', 
            name='Citations', 
            line=dict(color='red', width=3)
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Publications", row=1, col=1)
    fig.update_yaxes(title_text="Citations", row=2, col=1)
    fig.update_layout(height=600, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Collaboration analysis
    st.subheader(f"{country}'s Collaboration Profile")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['avg_collaboration'],
            title={'text': "Collaboration Index"},
            gauge={
                'axis': {'range': [None, 2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.8], 'color': "lightgray"},
                    {'range': [0.8, 1.2], 'color': "gray"},
                    {'range': [1.2, 2], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4}, 
                    'thickness': 0.75, 
                    'value': 1.5
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        doc_categories = country_data[
            ['% Documents in Top 1%', '% Documents in Top 10%']
        ].mean()
        fig = go.Figure(data=[go.Bar(
            x=['Top 1%', 'Top 10%'],
            y=doc_categories.values,
            marker_color=['gold', 'silver']
        )])
        fig.update_layout(title="Document Quality Distribution (%)", height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Comparison with global average
    st.subheader(f"{country} vs Global Average")
    
    metrics_list = ['Web of Science Documents', 'Times Cited', 'Collab-CNCI', '% Docs Cited']
    comparison_df = processor.compare_country_to_global(country, metrics_list)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=country, 
        x=comparison_df['Metric'], 
        y=comparison_df[country]
    ))
    fig.add_trace(go.Bar(
        name='Global Average', 
        x=comparison_df['Metric'], 
        y=comparison_df['Global Average']
    ))
    fig.update_layout(barmode='group', height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_temporal_trends(processor, filtered_df):
    """Render the Temporal Trends section"""
    st.header("Temporal Trends Analysis")
    
    yearly_data = processor.get_temporal_aggregates(filtered_df)
    
    # Multi-metric trend
    st.subheader("Global Research Metrics Over Time")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            'Total Publications', 
            'Total Citations', 
            'Average Collaboration Index'
        ),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'], 
            y=yearly_data['Publications'],
            fill='tozeroy', 
            name='Publications', 
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'], 
            y=yearly_data['Citations'],
            fill='tozeroy', 
            name='Citations', 
            line=dict(color='red')
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=yearly_data['Year'], 
            y=yearly_data['Avg_Collaboration'],
            mode='lines+markers', 
            name='Collaboration', 
            line=dict(color='green')
        ),
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Year", row=3, col=1)
    fig.update_layout(height=900, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth rate analysis
    st.subheader("Year-over-Year Growth Rates")
    
    yearly_data = processor.calculate_growth_rates(yearly_data)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Publication Growth %', 
        x=yearly_data['Year'], 
        y=yearly_data['Pub_Growth']
    ))
    fig.add_trace(go.Bar(
        name='Citation Growth %', 
        x=yearly_data['Year'], 
        y=yearly_data['Citation_Growth']
    ))
    fig.update_layout(
        barmode='group', 
        height=400, 
        title="Year-over-Year Growth Rates (%)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Publication Heatmap: Countries vs Years")
    
    pivot_data = processor.create_publication_heatmap_data(filtered_df, top_n=20)
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Year", y="Country", color="Publications"),
        aspect="auto",
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_collaboration_patterns(processor, filtered_df):
    """Render the Collaboration Patterns section"""
    st.header("Collaboration Network Analysis")
    
    # Distribution
    st.subheader("Distribution of Collaboration Indices")
    
    fig = px.histogram(
        filtered_df,
        x='Collab-CNCI',
        nbins=50,
        title="Collaboration Index Distribution",
        labels={'Collab-CNCI': 'Collaboration Index'},
        color_discrete_sequence=['steelblue']
    )
    fig.add_vline(
        x=filtered_df['Collab-CNCI'].mean(), 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Mean"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation - REMOVED TRENDLINE
    st.subheader("🔗 Collaboration Impact on Citations")
    
    fig = px.scatter(
        filtered_df,
        x='Collab-CNCI',
        y='Times Cited',
        size='Web of Science Documents',
        color='Name',
        hover_data=['Name', 'year'],
        title="Collaboration Index vs Citations (bubble size = publications)"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    correlation = processor.calculate_collaboration_correlation(filtered_df)
    st.markdown(f"**Correlation coefficient:** {correlation:.3f}")
    
    # Top collaborators
    st.subheader("Top Collaborative Countries")
    
    top_collab = filtered_df.nlargest(20, 'Collab-CNCI')[
        ['Name', 'Collab-CNCI', 'Times Cited', 'year']
    ]
    
    fig = px.scatter(
        top_collab,
        x='Collab-CNCI',
        y='Times Cited',
        size='Collab-CNCI',
        color='Name',
        text='Name',
        title="Top 20 Countries by Collaboration Index"
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional patterns
    st.subheader("Collaboration Patterns by Region")
    
    col1, col2 = st.columns(2)
    
    with col1:
        avg_collab = filtered_df.groupby('Name')['Collab-CNCI'].mean().sort_values(
            ascending=False
        ).head(15)
        fig = px.bar(
            x=avg_collab.values,
            y=avg_collab.index,
            orientation='h',
            title="Average Collaboration Index by Country (Top 15)",
            labels={'x': 'Avg Collaboration Index', 'y': 'Country'},
            color=avg_collab.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        avg_docs_cited = filtered_df.groupby('Name')['% Docs Cited'].mean().sort_values(
            ascending=False
        ).head(15)
        fig = px.bar(
            x=avg_docs_cited.values,
            y=avg_docs_cited.index,
            orientation='h',
            title="Average % Documents Cited (Top 15)",
            labels={'x': '% Docs Cited', 'y': 'Country'},
            color=avg_docs_cited.values,
            color_continuous_scale='Plasma'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def render_advanced_insights(processor, filtered_df):
    """Render the Advanced Insights section"""
    st.header("Advanced Analytics & Insights")
    
    # Statistical summary
    st.subheader("Statistical Summary")
    summary_stats = processor.get_statistical_summary(filtered_df)
    st.dataframe(summary_stats.style.highlight_max(axis=0))
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    corr_matrix = processor.calculate_correlation_matrix(filtered_df)
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Matrix"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Outlier detection
    st.subheader("Outlier Analysis")
    outliers = processor.detect_outliers(filtered_df, 'Times Cited', 2.0)
    
    st.write("**Countries with unusually high citation counts (Z-score > 2):**")
    st.dataframe(outliers.head(10))
    
    # Performance quadrants
    st.subheader("Performance Quadrants")
    quadrant_df = processor.create_performance_quadrants(filtered_df)
    
    fig = px.scatter(
        quadrant_df,
        x='Web of Science Documents',
        y='Times Cited',
        color='Quadrant',
        hover_data=['Name', 'year'],
        title="Performance Quadrants: Publications vs Citations",
        size='Collab-CNCI'
    )
    
    fig.add_hline(
        y=quadrant_df['Median_Citations'].iloc[0], 
        line_dash="dash", 
        line_color="gray"
    )
    fig.add_vline(
        x=quadrant_df['Median_Pubs'].iloc[0], 
        line_dash="dash", 
        line_color="gray"
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Predictive insights
    st.subheader("🔮 Predictive Insights")
    st.markdown("""
    <div class="insight-box">
    <b>Key Findings:</b><br>
    1. <b>Collaboration drives impact:</b> Countries with higher collaboration indices show significantly better citation rates.<br>
    2. <b>Quality over quantity:</b> Some countries with moderate publication volumes achieve exceptional citation rates through strategic partnerships.<br>
    3. <b>Emerging trends:</b> Recent years show increased international collaboration, indicating a shift toward global research networks.<br>
    4. <b>Regional patterns:</b> Developed nations maintain leadership in citation efficiency, but emerging economies are rapidly closing the gap.
    </div>
    """, unsafe_allow_html=True)
    
    # Ranking dynamics
    st.subheader("Ranking Dynamics")
    rank_evolution = processor.get_ranking_dynamics(filtered_df, top_n=15)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rank_evolution.index,
        x=rank_evolution['mean'],
        orientation='h',
        name='Average Rank',
        error_x=dict(type='data', array=rank_evolution['std'])
    ))
    fig.update_layout(
        title="Top 15 Countries by Average Rank (with standard deviation)",
        xaxis_title="Rank (lower is better)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Data export
    st.subheader("Export Filtered Data")
    csv = processor.export_to_csv(filtered_df)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_publication_data.csv',
        mime='text/csv'
    )


def main():
    """Main application entry point"""
    try:
        # Initialize processor
        processor = get_data_processor()
        
        # Header
        st.markdown(
            '<p class="main-header">Global Publication Analytics Dashboard</p>', 
            unsafe_allow_html=True
        )
        st.markdown("---")
        
        # Sidebar filters
        st.sidebar.header("Filters & Controls")
        
        min_year, max_year = processor.get_year_range()
        year_range = st.sidebar.slider(
            "Select Year Range",
            min_year,
            max_year,
            (min_year, max_year)
        )
        
        countries = ['All'] + processor.get_available_countries()
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            countries,
            default=['All']
        )
        
        max_rank = processor.get_max_rank()
        rank_filter = st.sidebar.slider("Maximum Rank", 1, max_rank, max_rank)
        
        # Filter data
        filtered_df = processor.filter_data(year_range, selected_countries, rank_filter)
        
        # Navigation
        view = st.sidebar.radio(
            "📍 Navigate", 
            ["Overview", "Country Analysis", "Temporal Trends", 
             "Collaboration Patterns", "Advanced Insights"]
        )
        
        # Render selected view
        if view == "Overview":
            render_overview(processor, filtered_df)
        elif view == "Country Analysis":
            render_country_analysis(processor)
        elif view == "Temporal Trends":
            render_temporal_trends(processor, filtered_df)
        elif view == "Collaboration Patterns":
            render_collaboration_patterns(processor, filtered_df)
        elif view == "Advanced Insights":
            render_advanced_insights(processor, filtered_df)
            
    except FileNotFoundError:
        st.error("❌ Error: 'publications.csv' file not found. Please ensure the file is in the same directory as this script.")
        st.info("💡 Place your 'publications.csv' file in the same folder as this script and refresh the page.")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("Please check your data file format and try again.")


if __name__ == "__main__":
    main()