import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest, GradientBoostingRegressor
# Performance limits
MAX_STATS_MEASURES = 10  # Max measures to plot in Statistical Insights
MAX_ML_MEASURES = 10    # Max measures to consider in ML Insights
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
def generate_html_report(df, dimensions, measures, dates):
    
    """Generate a single colourful HTML report containing key tables and Plotly figures."""
    html_parts = []
    # Basic page header with styling
    html_parts.append("""
    <style>
        body {font-family: 'Arial', sans-serif; margin: 20px; background-color: #f9f9f9; color: #333;}
        h1, h2 {color: #1f77b4;}
        table {border-collapse: collapse; width: 100%; margin-bottom: 20px;}
        th, td {border: 1px solid #ddd; padding: 8px; text-align: left;}
        th {background-color: #667eea; color: white;}
        tr:nth-child(even) {background-color: #f2f2f2;}
    </style>
    """)
    html_parts.append("<h1>DataLens Pro Dashboard Export</h1>")

    # Data preview (first 100 rows)
    html_parts.append("<h2>Data Preview (first 100 rows)</h2>")
    html_parts.append(df.head(100).to_html(index=False, border=0, classes='table'))

    # KPI overview table for first few measures
    if len(measures) > 0:
        html_parts.append("<h2>KPI Overview</h2>")
        kpi_rows = []
        for measure in measures[:5]:
            total = df[measure].sum()
            avg = df[measure].mean()
            kpi_rows.append(f"<tr><td>{measure}</td><td>{total:,.2f}</td><td>{avg:,.2f}</td></tr>")
        kpi_table = "<table><tr><th>Measure</th><th>Total</th><th>Avg</th></tr>" + "".join(kpi_rows) + "</table>"
        html_parts.append(kpi_table)

    # Sample distribution plots for measures
    import plotly.express as px
    html_parts.append("<h2>Sample Distribution Plots</h2>")
    for measure in measures[:5]:
        fig = px.histogram(df, x=measure, nbins=30, title=f"{measure} Distribution")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Correlation heatmap if enough numeric columns
    if len(measures) >= 2:
        html_parts.append("<h2>Correlation Heatmap</h2>")
        corr = df[measures].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Heatmap")
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs='cdn'))

    # Assemble full HTML
    full_html = "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Dashboard Export</title></head><body>" + "\n".join(html_parts) + "</body></html>"
    return full_html

# Export button moved into main (see later in `main()`)

st.set_page_config(
    page_title="DataLens Pro - Tableau Replacement",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== STYLING ====================
st.markdown("""
<style>
   .main-header {font-size: 2.5rem; font-weight: 700; color: #1f77b4; margin-bottom: 0;}
   .sub-header {font-size: 1.1rem; color: #666; margin-top: 0;}
   .kpi-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
               padding: 1.5rem; border-radius: 10px; color: white; text-align: center;}
   .kpi-value {font-size: 2rem; font-weight: bold;}
   .kpi-label {font-size: 0.9rem; opacity: 0.9;}
    div[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
@st.cache_data
def load_data(file):
    """Load CSV or Excel with caching"""
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding_errors='ignore')
        elif file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file type. Upload CSV or Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def detect_column_types(df):
    """Auto-detect column types for Tableau-like Dimensions vs Measures"""
    dimensions, measures, dates = [], [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() < 20 and df[col].nunique() / len(df) < 0.05:
                dimensions.append(col)
            else:
                measures.append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            dates.append(col)
        else:
            # Attempt to parse as datetime with inference; if successful treat as date column
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            if not parsed.isna().all():
                df[col] = parsed
                dates.append(col)
            else:
                dimensions.append(col)
    return dimensions, measures, dates

# Cached correlation computation
@st.cache_data
def compute_corr(df, measures):
    return df[measures].corr()

def apply_global_filters(df, filters):
    """Apply sidebar filters to dataframe"""
    df_filtered = df.copy()
    for col, values in filters.items():
        if not values:
            continue
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            # Categorical filter expects a list of selected options
            df_filtered = df_filtered[df_filtered[col].isin(values)]
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            # Date filter expects a tuple/list of (start, end)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                start_ts = pd.to_datetime(values[0])
                end_ts = pd.to_datetime(values[1])
                df_filtered = df_filtered[(df_filtered[col] >= start_ts) & (df_filtered[col] <= end_ts)]
        else:
            # Numeric filter expects a tuple/list of (min, max)
            if isinstance(values, (list, tuple)) and len(values) == 2:
                df_filtered = df_filtered[(df_filtered[col] >= values[0]) & (df_filtered[col] <= values[1])]
    return df_filtered

# ==================== KPI DASHBOARD GENERATORS ====================
def generate_kpi_dashboard(df, measures, dimensions):
    """Dashboard 1-10: KPI Cards"""
    st.subheader("📈 KPI Overview Dashboard")
    if not measures:
        st.write("No numeric measures available for KPI calculation.")
        return
    num_cols = max(1, min(5, len(measures)))
    cols = st.columns(num_cols)
    for i, measure in enumerate(measures[:10]):
        with cols[i % num_cols]:
            total = df[measure].sum()
            avg = df[measure].mean()
            st.metric(
                label=measure,
                value=f"{total:,.2f}" if total > 1000 else f"{total:.2f}",
                delta=f"Avg: {avg:,.2f}"
            )

def generate_summary_stats_dashboard(df, measures):
    """Dashboard 11-15: Summary Statistics"""
    st.subheader("📊 Statistical Summary Dashboard")
    if measures:
        st.dataframe(df[measures].describe().T.style.background_gradient(cmap='Blues'), width='stretch')

def generate_missing_data_dashboard(df):
    """Dashboard 16: Missing Data Analysis"""
    st.subheader("🔍 Data Quality Dashboard - Missing Values")
    missing = df.isnull().sum().reset_index()
    missing.columns = ['Column', 'Missing Count']
    missing['Percent'] = (missing['Missing Count'] / len(df) * 100).round(2)
    missing = missing[missing['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    if not missing.empty:
        fig = px.bar(missing, x='Column', y='Percent', title='Missing Data % by Column',
                     color='Percent', color_continuous_scale='Reds')
        st.plotly_chart(fig, width='stretch')
    else:
        st.success("No missing data found!")

def generate_correlation_dashboard(df, measures):
    """Dashboard 17-20: Correlation Dashboards"""
    if len(measures) >= 2:
        st.subheader("🔗 Correlation Analysis Dashboard")
        corr = df[measures].corr()

        col1, col2 = st.columns(2)
        with col1:
            fig = px.imshow(corr, text_auto='.2f', aspect="auto",
                          title="Correlation Heatmap", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, width='stretch')
        with col2:
            fig = px.imshow(corr.abs(), text_auto='.2f', aspect="auto",
                          title="Absolute Correlation", color_continuous_scale='Viridis')
            st.plotly_chart(fig, width='stretch')

def generate_distribution_dashboard(df, measures, dimensions):
    """Dashboard 21-30: Distribution Dashboards"""
    st.subheader("📉 Distribution Dashboard")
    if not measures:
        st.info("No numeric measures available for distribution charts.")
        return
    tabs = st.tabs([f"Dist: {m}" for m in measures[:10]])
    for i, measure in enumerate(measures[:10]):
        with tabs[i]:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df, x=measure, marginal="box", nbins=50,
                                 title=f"Histogram of {measure}")
                st.plotly_chart(fig, width='stretch')
            with col2:
                fig = px.box(df, y=measure, points="outliers", title=f"Box Plot of {measure}")
                st.plotly_chart(fig, width='stretch')

def generate_categorical_dashboard(df, dimensions, measures):
    """Dashboard 31-40: Categorical Analysis"""
    st.subheader("🏷️ Categorical Analysis Dashboard")
    for dim in dimensions[:5]:
        if df[dim].nunique() < 50:
            with st.expander(f"Analysis by {dim}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    counts = df[dim].value_counts().reset_index()
                    counts.columns = [dim, 'count']
                    fig = px.bar(counts, x=dim, y='count', title=f"Count by {dim}")
                    st.plotly_chart(fig, width='stretch')
                with col2:
                    fig = px.pie(counts, names=dim, values='count', title=f"Distribution of {dim}")
                    st.plotly_chart(fig, width='stretch')

def generate_time_series_dashboard(df, dates, measures):
    """Dashboard 41-45: Time Series Dashboards"""
    if dates and measures:
        st.subheader("📅 Time Series Dashboard")
        date_col = dates[0]
        df_ts = df.copy()
        df_ts['period'] = df_ts[date_col].dt.to_period('M').astype(str)

        for measure in measures[:5]:
            agg = df_ts.groupby('period')[measure].agg(['sum', 'mean', 'count']).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=agg['period'], y=agg['sum'], name='Sum'), secondary_y=False)
            fig.add_trace(go.Scatter(x=agg['period'], y=agg['mean'], name='Avg'), secondary_y=True)
            fig.update_layout(title=f"{measure} Over Time", xaxis_title="Period")
            st.plotly_chart(fig, width='stretch')

def generate_scatter_matrix_dashboard(df, measures):
    """Dashboard 46-47: Scatter Matrix"""
    if len(measures) >= 3:
        st.subheader("🔢 Scatter Matrix Dashboard")
        fig = px.scatter_matrix(df[measures[:6]], dimensions=measures[:6],
                               title="Scatter Matrix of Top Measures")
        fig.update_traces(diagonal_visible=False)
        st.plotly_chart(fig, width='stretch')

def generate_pca_dashboard(df, measures):
    """Dashboard 48: PCA Analysis"""
    if len(measures) >= 3:
        st.subheader("🧬 PCA Dimensionality Dashboard")
        df_clean = df[measures].dropna()
        if len(df_clean) > 10:
            scaled = StandardScaler().fit_transform(df_clean)
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled)
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            fig = px.scatter(pca_df, x='PC1', y='PC2',
                           title=f"PCA - Explained Variance: {pca.explained_variance_ratio_.sum():.2%}")
            st.plotly_chart(fig, width='stretch')

def generate_cluster_dashboard(df, measures):
    """Dashboard 49: K-Means Clustering"""
    if len(measures) >= 2:
        st.subheader("🎯 Clustering Dashboard")
        n_clusters = st.slider("Number of Clusters", 2, 10, 3, key="cluster_slider")
        df_clean = df[measures[:3]].dropna()
        if len(df_clean) > n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df_clean['Cluster'] = kmeans.fit_predict(StandardScaler().fit_transform(df_clean))
            fig = px.scatter_3d(df_clean, x=measures[0], y=measures[1],
                               z=measures[2] if len(measures) > 2 else measures[0],
                               color='Cluster', title="K-Means Clusters")
            st.plotly_chart(fig, width='stretch')

def generate_custom_dashboard(df, dimensions, measures):
    """Dashboard 50+: Custom Builder"""
    st.subheader("🛠️ Custom Dashboard Builder")
    options_x = dimensions + measures
    if not options_x or not measures:
        st.info("Need at least one dimension and one measure for the custom dashboard.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("X-Axis", options_x, key="custom_x")
    with col2:
        y_axis = st.selectbox("Y-Axis", measures, key="custom_y")
    with col3:
        chart_type = st.selectbox("Chart Type",
                                  ["Bar", "Line", "Scatter", "Box", "Violin", "Area"], key="custom_chart")

    if x_axis and y_axis:
        if chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        elif chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
        elif chart_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        elif chart_type == "Violin":
            fig = px.violin(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
        else:
            fig = px.area(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        st.plotly_chart(fig, width='stretch')

# ==================== GRAPH GENERATORS - 50+ CHARTS ====================
def generate_all_graphs(df, dimensions, measures, dates):
    st.header("📊 Auto-Generated Graph Gallery - 50+ Charts")
    
    render_charts = st.checkbox("⚡ Enable Chart Rendering", value=False, key="render_auto_graphs")
    if not render_charts:
        st.info("Click the checkbox above to generate charts. This prevents slow initial tab loading.")
        return

    with st.spinner("Optimizing & rendering charts..."):
        cols = st.columns(2)
        graph_count = 0
        
        # Precompute once
        df_sorted = df.sort_values(dates[0]) if dates else df
        dim_uniques = {dim: df[dim].nunique() for dim in dimensions}
        
        def safe_plot(plot_func, *args, key, **kwargs):
            try:
                fig = plot_func(*args, **kwargs)
                st.plotly_chart(fig, use_container_width=True, key=key)
            except Exception as e:
                st.warning(f"Chart {key} failed: {str(e)}")
        
        # 1️⃣ Categorical Bar Charts
        with st.expander("Categorical Bar Charts", expanded=False):
            for dim in dimensions[:5]:
                if dim_uniques.get(dim, 0) < 30:
                    idx = graph_count % 2
                    graph_count += 1
                    with cols[idx]:
                        counts = df[dim].value_counts().head(20)
                        bar_df = pd.DataFrame({dim: counts.index, "count": counts.values})
                        safe_plot(px.bar, bar_df, x=dim, y="count",
                                  title=f"Graph {graph_count}: Count by {dim}", key=f"bar_{graph_count}")
        
        # 2️⃣ Distribution Analysis
        with st.expander("Distribution Analysis", expanded=False):
            for measure in measures[:10]:
                idx = graph_count % 2
                graph_count += 1
                with cols[idx]:
                    safe_plot(px.histogram, df, x=measure, nbins=30,
                              title=f"Graph {graph_count}: {measure} Distribution", key=f"hist_{graph_count}")
        
        # 3️⃣ Box Plot Analysis
        with st.expander("Box Plot Analysis", expanded=False):
            for measure in measures[:10]:
                idx = graph_count % 2
                graph_count += 1
                with cols[idx]:
                    safe_plot(px.box, df, y=measure,
                              title=f"Graph {graph_count}: {measure} Box Plot", key=f"box_{graph_count}")
        
        # 4️⃣ Correlation Scatter Plots
        with st.expander("Correlation Scatter Plots", expanded=False):
            for i in range(min(5, len(measures) - 1)):
                idx = graph_count % 2
                graph_count += 1
                with cols[idx]:
                    sub_df = df.sample(min(5000, len(df)), random_state=42)
                    safe_plot(px.scatter, sub_df, x=measures[i], y=measures[i+1], trendline="ols",
                              title=f"Graph {graph_count}: {measures[i]} vs {measures[i+1]}", key=f"scatter_{graph_count}")
        
        # 5️⃣ Time Series Line Charts (Aggregated)
        if dates:
            with st.expander("Time Series Line Charts", expanded=False):
                ts_agg = df_sorted.groupby(dates[0])[measures[:5]].mean().reset_index()
                for measure in measures[:5]:
                    idx = graph_count % 2
                    graph_count += 1
                    with cols[idx]:
                        safe_plot(px.line, ts_agg, x=dates[0], y=measure,
                                  title=f"Graph {graph_count}: {measure} Over Time", key=f"line_{graph_count}")
        
        # 6️⃣ Violin Plots
        with st.expander("Violin Plots", expanded=False):
            for measure in measures[:5]:
                idx = graph_count % 2
                graph_count += 1
                with cols[idx]:
                    safe_plot(px.violin, df, y=measure, box=True,
                              title=f"Graph {graph_count}: {measure} Violin", key=f"violin_{graph_count}")
        
        # 7️⃣ Heatmap Analysis
        with st.expander("Heatmap Analysis", expanded=False):
            for i in range(min(3, len(dimensions) - 1)):
                dim1, dim2 = dimensions[i], dimensions[i+1]
                if dim_uniques.get(dim1, 0) < 15 and dim_uniques.get(dim2, 0) < 15:
                    idx = graph_count % 2
                    graph_count += 1
                    with cols[idx]:
                        pivot = pd.crosstab(df[dim1], df[dim2])
                        safe_plot(px.imshow, pivot,
                                  title=f"Graph {graph_count}: {dim1} vs {dim2}", key=f"heat_{graph_count}")
        
        # 8️⃣ Area Chart Analysis
        if dates and measures:
            with st.expander("Area Chart Analysis", expanded=False):
                ts_agg = df_sorted.groupby(dates[0])[measures[:5]].sum().reset_index()
                for measure in measures[:5]:
                    idx = graph_count % 2
                    graph_count += 1
                    with cols[idx]:
                        safe_plot(px.area, ts_agg, x=dates[0], y=measure,
                                  title=f"Graph {graph_count}: {measure} Area", key=f"area_{graph_count}")
        
        # 9️⃣ Treemap Analysis
        with st.expander("Treemap Analysis", expanded=False):
            if measures:
                base_m = measures[0]
                for dim in dimensions[:5]:
                    if dim_uniques.get(dim, 0) < 40:
                        idx = graph_count % 2
                        graph_count += 1
                        with cols[idx]:
                            agg_df = df[[dim, base_m]].dropna().groupby(dim)[base_m].sum().reset_index()
                            safe_plot(px.treemap, agg_df, path=[dim], values=base_m,
                                      title=f"Graph {graph_count}: Treemap by {dim}", key=f"tree_{graph_count}")
        
        st.success(f"✅ Successfully rendered {graph_count} charts!")
def generate_ml_insights_dashboard(df, dimensions, measures, dates):
    st.subheader("🤖 Machine Learning Insights")
    if not measures:
        st.info("No measures available for ML.")
        return

    run_ml = st.checkbox("⚡ Compute ML Models", value=False, key="run_ml_insights")
    if not run_ml:
        st.info("Click to train models (takes ~2-5s). Prevents instant tab freeze.")
        return

    with st.spinner("Training ML models..."):
        target = measures[0]
        features = [m for m in measures[1:] if pd.api.types.is_numeric_dtype(df[m])]
        
        if len(features) < 2:
            st.warning("Need at least 2 measures for ML (1 target + 1 feature).")
            return

        # 🟢 SAMPLE DATA FOR FAST TRAINING & PLOTTING
        df_ml = df[features + [target]].dropna()
        if len(df_ml) < 50:
            st.warning("Not enough clean data for ML.")
            return
            
        df_ml = df_ml.sample(min(10000, len(df_ml)), random_state=42)
        X = df_ml[features]
        y = df_ml[target]

        # 1️⃣ Linear Regression
        st.markdown("### 📈 Linear Regression")
        lr = LinearRegression()
        lr.fit(X, y)
        y_pred = lr.predict(X)
        st.metric("R² Score (LR)", f"{r2_score(y, y_pred):.3f}")
        fig = px.scatter(x=y.values, y=y_pred, labels={"x":"Actual", "y":"Predicted"}, 
                         title="LR: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True, key="chart_lr_ml")

        # 2️⃣ Random Forest Importance
        st.markdown("### 🌲 Random Forest Feature Importance")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
        fig = px.bar(x=imp.index, y=imp.values, title="RF Feature Importance")
        st.plotly_chart(fig, use_container_width=True, key="chart_rf_ml")

        # 3️⃣ Gradient Boosting
        st.markdown("### 🚀 Gradient Boosting")
        gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr.fit(X, y)
        y_pred_gbr = gbr.predict(X)
        st.metric("R² Score (GBR)", f"{r2_score(y, y_pred_gbr):.3f}")

        # 4️⃣ K-Means Clustering
        st.markdown("### 🎯 K-Means Clustering")
        n_clusters = st.slider("Clusters", 2, 6, 3, key="ml_clusters_slider")
        from sklearn.preprocessing import StandardScaler
        X_scaled = StandardScaler().fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        clust_df = pd.DataFrame(X_scaled, columns=features)
        clust_df['Cluster'] = labels.astype(str)
        
        if len(features) >= 3:
            fig = px.scatter_3d(clust_df, x=features[0], y=features[1], z=features[2], 
                                color='Cluster', title="K-Means (3D)")
        else:
            fig = px.scatter(clust_df, x=features[0], y=features[1], 
                             color='Cluster', title="K-Means (2D)")
        st.plotly_chart(fig, use_container_width=True, key="chart_km_ml")

        # 5️⃣ Anomaly Detection
        st.markdown("### 🚨 Isolation Forest Anomaly Detection")
        iso = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso.fit_predict(X_scaled)
        cnt = pd.Series(anomalies).value_counts().rename({1:"Inlier", -1:"Outlier"})
        fig = px.pie(names=cnt.index, values=cnt.values, title="Anomaly Distribution")
        st.plotly_chart(fig, use_container_width=True, key="chart_anom_ml")

        # 6️⃣ ARIMA Forecast (Safe & Cached)
        if dates:
            st.markdown("### 📅 Time Series Forecast (ARIMA)")
            date_col = dates[0]
            try:
                ts = df[[date_col, target]].dropna().set_index(date_col)[target].sort_index().asfreq('D').ffill()
                if len(ts) > 50:
                    from statsmodels.tsa.arima.model import ARIMA
                    model = ARIMA(ts, order=(5,1,0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=30)
                    fig = px.line(x=forecast.index, y=forecast.values, title="30-Day ARIMA Forecast")
                    st.plotly_chart(fig, use_container_width=True, key="chart_arima_ml")
                else:
                    st.info("Not enough daily data for ARIMA.")
            except Exception as e:
                st.warning(f"ARIMA failed (usually due to missing dates or non-numeric index): {str(e)}")

        st.success("✅ ML Insights complete!")


def generate_statistical_insights_dashboard(df, dimensions, measures):
    """Dashboard with 50+ statistical insights and visualizations."""
    st.subheader("📊 Statistical Insights")
    if not measures:
        st.info("No measures available for statistical analysis.")
        return

    run_stats = st.checkbox("⚡ Compute Statistical Insights", value=False, key="run_stats_insights")
    if not run_stats:
        st.info("Click to run heavy statistical computations & plots.")
        return

    with st.spinner("Calculating statistics..."):
        # Fast descriptive table
        st.dataframe(df[measures].describe().T.style.background_gradient(cmap='Blues'), key="stats_desc_table")

        # Limit heavy plotting
        plot_measures = measures[:5]
        plot_dims = dimensions[:3]

        tabs = st.tabs([f"{m} Analysis" for m in plot_measures])
        for i, measure in enumerate(plot_measures):
            with tabs[i]:
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(df, x=measure, nbins=30, title=f"{measure} Histogram")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_hist_{i}")
                with col2:
                    fig = px.box(df, y=measure, title=f"{measure} Box Plot")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_box_{i}")

                col3, col4 = st.columns(2)
                with col3:
                    fig = px.violin(df, y=measure, box=True, title=f"{measure} Violin")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_violin_{i}")
                with col4:
                    import scipy.stats as stats
                    qq_data = df[measure].dropna()
                    if len(qq_data) > 4:
                        theoretical, sample = stats.probplot(qq_data, dist="norm", fit=False)
                        qq_df = pd.DataFrame({"theoretical": theoretical, "sample": sample})
                        fig = px.scatter(qq_df, x="theoretical", y="sample", trendline="ols",
                                         title=f"{measure} QQ Plot")
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_qq_{i}")

        # Correlation heatmap
        if len(measures) >= 2:
            corr = df[measures].corr()
            fig = px.imshow(corr, text_auto='.2f', aspect="auto", title="Correlation Heatmap")
            st.plotly_chart(fig, use_container_width=True, key="chart_corr_stat")

        # ANOVA (limited)
        with st.expander("ANOVA Tests (Categorical vs Measure)", expanded=False):
            for dim in plot_dims[:3]:
                if df[dim].nunique() < 10:
                    for meas in plot_measures[:2]:
                        try:
                            import statsmodels.api as sm
                            from statsmodels.formula.api import ols
                            model = ols(f"{meas} ~ C({dim})", data=df).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            st.markdown(f"**{meas} ~ {dim}**")
                            st.dataframe(anova_table.style.background_gradient(cmap='Greys'), key=f"anova_{dim}_{meas}")
                        except Exception:
                            pass

        st.success("✅ Statistical insights generated.")

def run_offline_parser(df, query, dimensions, measures):
    import re
    query_lower = query.lower().strip()
    
    # 1. Column matching helper
    def find_columns(text, cols):
        matched = []
        for col in cols:
            if col.lower() in text:
                matched.append(col)
        # Sort by length descending to match longer names first
        return sorted(matched, key=len, reverse=True)
        
    matched_measures = find_columns(query_lower, measures)
    matched_dims = find_columns(query_lower, dimensions)
    
    # 2. Detect operations
    operation = None
    if any(word in query_lower for word in ["average", "mean", "avg"]):
        operation = "mean"
    elif any(word in query_lower for word in ["sum", "total", "add"]):
        operation = "sum"
    elif any(word in query_lower for word in ["count", "number of", "how many"]):
        operation = "count"
    elif any(word in query_lower for word in ["max", "highest", "maximum"]):
        operation = "max"
    elif any(word in query_lower for word in ["min", "lowest", "minimum"]):
        operation = "min"
        
    # Case A: Aggregation by Group (e.g. "average sales by region" or "sum profit grouped by segment")
    if operation and matched_measures and matched_dims:
        meas = matched_measures[0]
        dim = matched_dims[0]
        
        st.write(f"📊 **Offline Interpreter output:** Calculating the **{operation}** of **{meas}** grouped by **{dim}**.")
        try:
            if operation == "mean":
                res = df.groupby(dim)[meas].mean().reset_index()
            elif operation == "sum":
                res = df.groupby(dim)[meas].sum().reset_index()
            elif operation == "count":
                res = df.groupby(dim)[meas].count().reset_index()
            elif operation == "max":
                res = df.groupby(dim)[meas].max().reset_index()
            elif operation == "min":
                res = df.groupby(dim)[meas].min().reset_index()
                
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(res.style.background_gradient(subset=[meas], cmap="Blues"), use_container_width=True)
            with col2:
                fig = px.bar(res, x=dim, y=meas, title=f"{meas} ({operation}) by {dim}", color=meas, color_continuous_scale="Blues")
                st.plotly_chart(fig, use_container_width=True, key="offline_parser_bar")
        except Exception as e:
            st.error(f"Could not group data: {str(e)}")
            
    # Case B: General Filtering (e.g. "show rows where Profit > 500" or "filter where age < 30")
    elif "where" in query_lower or "filter" in query_lower:
        # Detect column, operator, value
        pattern = r"(\w+)\s*(>|<|==|>=|<=)\s*([\d\.]+)"
        match = re.search(pattern, query_lower)
        if match:
            col_name, op, val = match.groups()
            # Match col_name against actual columns
            actual_col = None
            for col in df.columns:
                if col.lower() == col_name:
                    actual_col = col
                    break
            
            if actual_col:
                try:
                    val_num = float(val)
                    st.write(f"🔍 **Offline Interpreter output:** Filtering rows where **{actual_col}** {op} {val_num}.")
                    if op == ">":
                        res = df[df[actual_col] > val_num]
                    elif op == "<":
                        res = df[df[actual_col] < val_num]
                    elif op == "==":
                        res = df[df[actual_col] == val_num]
                    elif op == ">=":
                        res = df[df[actual_col] >= val_num]
                    elif op == "<=":
                        res = df[df[actual_col] <= val_num]
                        
                    st.metric("Rows found", f"{len(res):,}")
                    st.dataframe(res.head(100), use_container_width=True)
                except Exception as e:
                    st.error(f"Filter failed: {str(e)}")
            else:
                st.warning(f"Could not identify the column '{col_name}' in your dataset.")
        else:
            st.warning("For filtering, please use the format: `column > value` or `column < value` (e.g. `Profit > 500`).")
            
    # Case C: Single measure statistics (e.g. "average sales" or "max profit")
    elif operation and matched_measures:
        meas = matched_measures[0]
        st.write(f"📊 **Offline Interpreter output:** Calculating the global **{operation}** of **{meas}**.")
        try:
            if operation == "mean":
                val = df[meas].mean()
            elif operation == "sum":
                val = df[meas].sum()
            elif operation == "count":
                val = df[meas].count()
            elif operation == "max":
                val = df[meas].max()
            elif operation == "min":
                val = df[meas].min()
                
            st.metric(label=f"{meas} ({operation.capitalize()})", value=f"{val:,.2f}" if val > 100 else f"{val:.4f}")
        except Exception as e:
            st.error(f"Stat computation failed: {str(e)}")
            
    else:
        st.write("🤖 **Offline Interpreter:** I didn't fully match your query schema. Try queries like:")
        st.write("- `average Sales by Segment` (Grouped aggregation)")
        st.write("- `sum Profit grouped by Region` (Grouped aggregation)")
        st.write("- `filter where Profit > 1000` (Filtering rows)")
        st.write("- `max Discount` (Simple metric computation)")


def generate_more_stats_dashboard(df, dimensions, measures, dates):
    st.subheader("📈 More Stats: AutoML & Smart Query Workbench")

    ai_tab, automl_tab = st.tabs(["🤖 PandasAI (Smart Query)", "🚀 AutoML Workbench"])

    with ai_tab:
        st.markdown("### 🤖 Chat with your Data")
        st.write("Ask questions in natural language and get instant analysis, data tables, and charts.")

        pandasai_installed = False
        try:
            import pandasai  # noqa: F401
            pandasai_installed = True
        except ImportError:
            pass

        if pandasai_installed:
            st.info("💡 **PandasAI is installed!** You can use standard LLM-powered queries.")
            api_key = st.text_input(
                "Enter OpenAI API Key (optional - leave blank to use Offline Smart Parser)",
                type="password",
                key="pandasai_api_key"
            )
            query = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g. 'Show me the average sales grouped by region'",
                key="pandasai_query"
            )

            if query:
                if api_key:
                    with st.spinner("PandasAI is thinking..."):
                        try:
                            from pandasai import SmartDataframe
                            from pandasai.llm import OpenAI
                            llm = OpenAI(api_token=api_key)
                            sdf = SmartDataframe(df, config={"llm": llm})
                            result = sdf.chat(query)
                            st.write(result)
                        except Exception as e:
                            st.error(f"PandasAI Error: {str(e)}")
                else:
                    run_offline_parser(df, query, dimensions, measures)
        else:
            st.info("💡 Running in **Offline Smart Parser Mode** (zero setup required).")
            query = st.text_input(
                "Ask a question about your data:",
                placeholder="e.g. 'average sales by region' or 'show rows where Profit > 500'",
                key="offline_query"
            )
            if query:
                run_offline_parser(df, query, dimensions, measures)

    with automl_tab:
        st.markdown("### 🚀 AutoML Leaderboard")
        st.write(
            "Select one or more target variables and predictor features. "
            "DataLens will automatically train, tune, and evaluate multiple regression algorithms."
        )

        if not measures:
            st.info("No measures available for machine learning.")
            return

        target_vars = st.multiselect(
            "Target Variable(s)",
            measures,
            default=measures[:1],
            key="automl_targets"
        )

        predictor_pool = [c for c in (measures + dimensions) if c not in target_vars]
        features = st.multiselect(
            "Predictor Features",
            predictor_pool,
            default=predictor_pool[:4] if predictor_pool else [],
            key="automl_features"
        )

        run_automl = st.button("⚡ Run AutoML Pipeline", key="run_automl")

        if not run_automl:
            return

        if not target_vars:
            st.warning("Please select at least one target variable.")
            return

        if not features:
            st.warning("Please select at least one predictor feature.")
            return

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import Ridge, Lasso, ElasticNet
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import time

        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "ElasticNet Regression": ElasticNet(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "KNeighbors Regressor": KNeighborsRegressor(n_neighbors=5),
            "Support Vector Regressor": SVR(kernel="rbf"),
            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "AdaBoost Regressor": AdaBoostRegressor(n_estimators=100, random_state=42),
            "MLP Neural Network": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
            "Bayesian Ridge": BayesianRidge(),
            "Huber Regressor": HuberRegressor(),
            "PLS Regression": "dynamic",
            "Gaussian Process Regressor": GaussianProcessRegressor()
        }

        # Keep training manageable for heavy models
        for target in target_vars:
            st.markdown(f"## 🎯 Target: {target}")

            df_clean = df[[target] + features].dropna()
            if len(df_clean) < 30:
                st.warning(f"Not enough clean rows for target '{target}' (need at least 30 non-null rows).")
                continue

            # Smaller sample keeps GPR and MLP responsive
            if len(df_clean) > 2000:
                df_clean = df_clean.sample(2000, random_state=42)

            X = pd.get_dummies(df_clean[features], drop_first=True)
            y = pd.to_numeric(df_clean[target], errors="coerce")

            valid_mask = y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            if len(X) < 30:
                st.warning(f"Target '{target}' became too small after numeric cleanup.")
                continue

            if X.shape[1] == 0:
                st.warning(f"No usable predictor columns remained after encoding for target '{target}'.")
                continue

            if y.nunique() < 2:
                st.warning(f"Target '{target}' has only one unique value after cleaning.")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            leaderboard = []

            with st.spinner(f"Training models for {target}..."):
                for name, model in models.items():
                    try:
                        start_time = time.time()

                        if name == "PLS Regression":
                            # For a single target regression problem, PLS must use 1 component.
                            n_comp = min(1, X_train.shape[1], max(1, X_train.shape[0] - 1))
                            model = PLSRegression(n_components=max(1, n_comp))

                        model.fit(X_train, y_train)
                        elapsed = time.time() - start_time

                        preds = model.predict(X_test)
                        preds = np.asarray(preds).ravel()

                        r2 = r2_score(y_test, preds)
                        mae = mean_absolute_error(y_test, preds)
                        rmse = np.sqrt(mean_squared_error(y_test, preds))

                        leaderboard.append({
                            "Algorithm": name,
                            "R² Score": r2,
                            "Mean Absolute Error (MAE)": mae,
                            "Root Mean Squared Error (RMSE)": rmse,
                            "Training Time (s)": elapsed
                        })

                    except Exception as e:
                        st.warning(f"Skipping {name} for target '{target}' due to error: {e}")
                        continue

            if not leaderboard:
                st.warning(f"No models could be trained successfully for target '{target}'.")
                continue

            ld_df = (
                pd.DataFrame(leaderboard)
                .sort_values(by="R² Score", ascending=False)
                .reset_index(drop=True)
            )

            best_model_name = ld_df.iloc[0]["Algorithm"]
            best_r2 = ld_df.iloc[0]["R² Score"]

            st.success(f"🏆 **Winner for {target}: {best_model_name}** with R² score of **{best_r2:.3f}**!")

            st.dataframe(
                ld_df.style
                .background_gradient(subset=["R² Score"], cmap="Greens")
                .background_gradient(subset=["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"], cmap="Reds"),
                use_container_width=True
            )

            fig = px.bar(
                ld_df,
                x="Algorithm",
                y="R² Score",
                color="R² Score",
                color_continuous_scale="Viridis",
                title=f"Model Performance Leaderboard for {target} (R² Score)"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"automl_leaderboard_chart_{target}")


def main():
    st.markdown('<p class="main-header">DataLens Pro 📊</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Open-Source Tableau Replacement - Upload data, get 50+ dashboards instantly</p>',
                unsafe_allow_html=True)

    # Sidebar - File Upload
    with st.sidebar:
        st.header("1. Upload Data")
        uploaded_files = st.file_uploader("Choose CSV or Excel files",
                                         type=['csv', 'xlsx', 'xls'],
                                         accept_multiple_files=True)

        st.header("2. Global Filters")
        st.caption("Filters apply to all dashboards")

    if uploaded_files:
        # Load and combine data
        dfs = []
        for file in uploaded_files:
            df = load_data(file)
            if df is not None:
                dfs.append(df)

        if not dfs:
            st.stop()

        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # Detect types
        dimensions, measures, dates = detect_column_types(df)

        # Sidebar filters
        filters = {}
        with st.sidebar:
            st.subheader("Dimensions")
            for dim in dimensions[:8]:
                if df[dim].nunique() < 100:
                    options = st.multiselect(f"{dim}", df[dim].dropna().unique(), key=f"filter_{dim}")
                    if options:
                        filters[dim] = options

            st.subheader("Measures")
            for measure in measures[:5]:
                min_val, max_val = float(df[measure].min()), float(df[measure].max())
                val_range = st.slider(f"{measure}", min_val, max_val, (min_val, max_val), key=f"filter_{measure}")
                if val_range!= (min_val, max_val):
                    filters[measure] = val_range

            if dates:
                st.subheader("Date Range")
                for date_col in dates[:2]:
                    min_date, max_date = df[date_col].min(), df[date_col].max()
                    date_range = st.date_input(f"{date_col}", [min_date, max_date], key=f"filter_{date_col}")
                    if len(date_range) == 2:
                        filters[date_col] = date_range

        # Apply filters
        df_filtered = apply_global_filters(df, filters)

        # HTML Export button
        if st.sidebar.button("Export Dashboard (HTML)"):
            html_report = generate_html_report(df_filtered, dimensions, measures, dates)
            st.sidebar.download_button(
                label="Download HTML",
                data=html_report,
                file_name="dashboard.html",
                mime="text/html"
            )

        # Data Preview
        with st.expander("📋 Data Preview & Info", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Rows", f"{len(df_filtered):,}")
            col2.metric("Total Columns", len(df_filtered.columns))
            col3.metric("Measures", len(measures))
            col4.metric("Dimensions", len(dimensions))
            st.dataframe(df_filtered.head(100), width='stretch')

            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Filtered Data", csv, "filtered_data.csv", "text/csv")

        # Tabs for all dashboards
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "🎯 KPI & Summary",
            "📊 Distributions",
            "🔗 Correlations",
            "📅 Time Series",
            "🧬 Advanced Analytics",
            "📈 50+ Auto Graphs",
            "🤖 ML Insights",
            "📚 Statistical & ML Insights",
            "📈 More Stats"
        ])

        with tab1:
            generate_kpi_dashboard(df_filtered, measures, dimensions)
            generate_summary_stats_dashboard(df_filtered, measures)
            generate_missing_data_dashboard(df_filtered)

        with tab2:
            generate_distribution_dashboard(df_filtered, measures, dimensions)
            generate_categorical_dashboard(df_filtered, dimensions, measures)

        with tab3:
            generate_correlation_dashboard(df_filtered, measures)
            generate_scatter_matrix_dashboard(df_filtered, measures)

        with tab4:
            generate_time_series_dashboard(df_filtered, dates, measures)

        with tab5:
            generate_pca_dashboard(df_filtered, measures)
            generate_cluster_dashboard(df_filtered, measures)
            generate_custom_dashboard(df_filtered, dimensions, measures)
        with tab6:
            generate_all_graphs(df_filtered, dimensions, measures, dates)

        with tab7:
            generate_ml_insights_dashboard(df_filtered, dimensions, measures, dates)
        with tab8:
            generate_statistical_insights_dashboard(df_filtered, dimensions, measures)
            # Removed duplicate ML call from tab8
        with tab9:
            generate_more_stats_dashboard(df_filtered, dimensions, measures, dates)

    else:
        st.info("👆 Upload a CSV or Excel file to start building dashboards")
        st.markdown("""
        ### How it works:
        1. **Upload** your CSV/Excel files - multiple files will be combined
        2. **Auto-Analysis** - App detects dimensions, measures, and dates
        3. **50+ Dashboards** - KPI, distributions, correlations, time series, clustering, and more
        4. **50+ Graphs** - Bar, line, scatter, box, violin, heatmap, treemap, etc
        5. **Filter** - Use sidebar to slice data across all charts
        6. **Export** - Download filtered data or charts

        ### Unlike Tableau:
        - ✅ 100% Free & Open Source
        - ✅ Python-powered - extend with any library
        - ✅ No row limits
        - ✅ Deploy anywhere with `streamlit run`
        """)

if __name__ == "__main__":
    main()