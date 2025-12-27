"""
Driver Analysis Dashboard

A Streamlit-based visualization dashboard for driver behavior analysis.

Usage:
    streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.scorer import DriverScorer, ScoreBreakdown


# Page config
st.set_page_config(
    page_title="Driver Analysis Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .score-display {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    .grade-A { color: #00C853; }
    .grade-B { color: #64DD17; }
    .grade-C { color: #FFD600; }
    .grade-D { color: #FF9100; }
    .grade-F { color: #FF1744; }
</style>
""", unsafe_allow_html=True)


def load_data(path: str) -> pd.DataFrame:
    """Load driving data from CSV."""
    return pd.read_csv(path)


def render_score_card(score: ScoreBreakdown):
    """Render the overall score card."""
    scorer = DriverScorer()
    grade = scorer.score_to_grade(score.overall_score)
    risk = scorer.score_to_risk_level(score.overall_score)
    
    # Color based on grade
    grade_colors = {
        'A': '#00C853', 'B': '#64DD17', 'C': '#FFD600', 
        'D': '#FF9100', 'F': '#FF1744'
    }
    color = grade_colors.get(grade, '#667eea')
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; 
                    background: linear-gradient(135deg, {color}40 0%, {color}20 100%);
                    border-radius: 15px; border: 2px solid {color};">
            <h1 style="margin:0; font-size: 72px; color: {color};">{score.overall_score}</h1>
            <h2 style="margin:0; color: {color};">Grade: {grade}</h2>
            <p style="margin:10px 0 0 0; font-size: 18px;">{risk}</p>
        </div>
        """, unsafe_allow_html=True)


def render_score_breakdown(score: ScoreBreakdown):
    """Render score breakdown as gauge charts."""
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}] * 4],
        subplot_titles=("Behavior", "Smoothness", "Awareness", "Speed")
    )
    
    scores = [
        ("Behavior", score.behavior_score),
        ("Smoothness", score.smoothness_score),
        ("Awareness", score.awareness_score),
        ("Speed", score.speed_score)
    ]
    
    for i, (name, value) in enumerate(scores):
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value,
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#667eea"},
                    'steps': [
                        {'range': [0, 40], 'color': "#ffebee"},
                        {'range': [40, 70], 'color': "#fff3e0"},
                        {'range': [70, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 2},
                        'thickness': 0.75,
                        'value': 60
                    }
                }
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)


def render_timeline(df: pd.DataFrame):
    """Render driving behavior timeline."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Speed (m/s)", "Steering (degrees)", "Behavior Label"),
        vertical_spacing=0.1
    )
    
    # Speed
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['speed'], mode='lines', 
                   name='Speed', line=dict(color='#667eea')),
        row=1, col=1
    )
    
    # Steering
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['steering'], mode='lines',
                   name='Steering', line=dict(color='#764ba2')),
        row=2, col=1
    )
    
    # Labels as colors
    if 'label' in df.columns:
        label_colors = {'safe': '#00C853', 'aggressive': '#FF1744', 'drowsy': '#FF9100'}
        colors = df['label'].map(label_colors)
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=[1]*len(df), mode='markers',
                       marker=dict(color=colors, size=5),
                       name='Behavior'),
            row=3, col=1
        )
    
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_label_distribution(df: pd.DataFrame):
    """Render label distribution pie chart."""
    if 'label' not in df.columns:
        return
    
    label_counts = df['label'].value_counts()
    
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Behavior Distribution",
        color_discrete_sequence=['#FF1744', '#FF9100', '#00C853'],
        hole=0.4
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_feature_distributions(df: pd.DataFrame):
    """Render feature distribution histograms."""
    features = ['speed', 'steering', 'steering_jerk', 'speed_change']
    available = [f for f in features if f in df.columns]
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=available)
    
    for i, feat in enumerate(available):
        row = i // 2 + 1
        col = i % 2 + 1
        fig.add_trace(
            go.Histogram(x=df[feat], name=feat, marker_color='#667eea'),
            row=row, col=col
        )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_model_comparison():
    """Render model comparison chart."""
    models = ['SVM', 'Random Forest', 'LSTM', 'Transformer']
    accuracies = [78.9, 96.6, 96.0, 96.9]
    f1_scores = [82.2, 96.6, 95.9, 96.6]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracies, 
                         marker_color='#667eea'))
    fig.add_trace(go.Bar(name='F1 Score', x=models, y=f1_scores,
                         marker_color='#764ba2'))
    
    fig.update_layout(
        title="Model Performance Comparison",
        yaxis_title="Score (%)",
        barmode='group',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


def get_contract_decision(score: float) -> tuple:
    """Get contract decision based on score."""
    if score >= 85:
        return "🟢 BONUS ELIGIBLE", "#00C853", "Performance bonus, recognition"
    elif score >= 70:
        return "🔵 RETAIN CONTRACT", "#2196F3", "Standard renewal"
    elif score >= 50:
        return "🟠 MANDATORY TRAINING", "#FF9100", "Safety training, 30-day review"
    else:
        return "🔴 TERMINATE CONTRACT", "#FF1744", "HR review, contract termination"


def generate_pdf_report(df, driver_id, driver_name):
    """Generate PDF report and return bytes."""
    from src.models.report_generator import DriverReportGenerator
    import tempfile
    import os
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = DriverReportGenerator(tmpdir)
        pdf_path = generator.generate_report(df, driver_id, driver_name)
        
        # Read the PDF as bytes
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
    
    return pdf_bytes


def main():
    st.title("🚗 Driver Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Driver Info
    st.sidebar.subheader("Driver Information")
    driver_id = st.sidebar.text_input("Driver ID", value="DRV_2024_001")
    driver_name = st.sidebar.text_input("Driver Name", value="Oualid Choukrad")
    
    st.sidebar.markdown("---")
    
    # Data source
    data_files = list(Path("data/processed").glob("*.csv"))
    if data_files:
        selected_file = st.sidebar.selectbox(
            "Select Data File",
            data_files,
            format_func=lambda x: x.name
        )
    else:
        st.sidebar.warning("No data files found")
        selected_file = None
    
    if selected_file is None:
        st.info("Please select a data file from the sidebar")
        return
    
    # Load data
    df = load_data(selected_file)
    
    # Compute score
    scorer = DriverScorer()
    score = scorer.compute_score(df)
    
    # Contract decision
    decision, decision_color, decision_action = get_contract_decision(score.overall_score)
    
    # Overall Score
    st.header("📊 Driver Safety Score")
    render_score_card(score)
    
    # Contract Decision Box
    st.markdown(f"""
    <div style="text-align: center; padding: 15px; margin: 20px 0;
                background: {decision_color}20; border: 2px solid {decision_color};
                border-radius: 10px;">
        <h2 style="margin:0; color: {decision_color};">{decision}</h2>
        <p style="margin: 5px 0 0 0;">{decision_action}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # PDF Download Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Generate Report")
    
    if st.sidebar.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_report(df, driver_id, driver_name)
                st.sidebar.download_button(
                    label="📥 Download PDF",
                    data=pdf_bytes,
                    file_name=f"driver_report_{driver_id}.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("Report ready! Click Download above.")
            except Exception as e:
                st.sidebar.error(f"Error generating report: {e}")
    
    st.markdown("---")
    
    # Score Breakdown
    st.header("📈 Score Breakdown")
    render_score_breakdown(score)
    
    # Risk Factors and Recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Risk Factors")
        if score.risk_factors:
            for risk in score.risk_factors:
                st.warning(risk)
        else:
            st.success("No significant risk factors detected!")
    
    with col2:
        st.subheader("💡 Recommendations")
        for rec in score.recommendations:
            st.info(rec)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Timeline", "Statistics", "Models"])
    
    with tab1:
        st.subheader("Driving Timeline")
        render_timeline(df.head(1000))  # Limit for performance
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            render_label_distribution(df)
        with col2:
            render_feature_distributions(df)
        
        # Summary stats
        st.subheader("Summary Statistics")
        stats_cols = ['speed', 'steering', 'steering_jerk', 'radar_distance']
        available_stats = [c for c in stats_cols if c in df.columns]
        st.dataframe(df[available_stats].describe().round(2))
    
    with tab3:
        st.subheader("Model Performance Comparison")
        render_model_comparison()
        
        st.markdown("""
        ### Model Details
        - **SVM**: Support Vector Machine with RBF kernel
        - **Random Forest**: 100 trees, max depth 15
        - **LSTM**: 2-layer bidirectional, 64 hidden units, 20-step sequences
        - **Transformer**: 2 encoder layers, 4 attention heads, 64 dim
        """)


if __name__ == "__main__":
    main()
