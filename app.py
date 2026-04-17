import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
import joblib

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="NYC Smart Restaurant Analytics", layout="wide")

# ------------------- DATA LOADING (with encoding fallback) -------------------
@st.cache_data
def load_data():
    # 加载由 df_with_violation_topic.ipynb 导出的数据集
    df = pd.read_csv("df_with_violation_topic.csv")
    # 过滤无效的行政区数据 (boro=0)
    df = df[df['boro'].astype(str) != '0']
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    return df

@st.cache_resource
def load_model():
    model_path = 'restaurant_closure_model_fixed.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Load data and model
df = load_data()
model = load_model()

# ------------------- SIDEBAR NAVIGATION -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "City Overview",
    "Violation Deep Dive",
    "Closure Risk Prediction"
])
# --- 侧边栏筛选器 (Sidebar Filters) ---
st.sidebar.header("Filters")

# Borough (行政区) 筛选器 - 使用正确的列名 'boro'
boroughs = st.sidebar.multiselect(
    "Select Borough(s):",
    options=df['boro'].unique(),
    default=df['boro'].unique()  # 默认全选
)

# Cuisine (菜系) 筛选器
cuisines = st.sidebar.multiselect(
    "Select Cuisine(s):",
    options=df['cuisine_description'].unique(),
    default=df['cuisine_description'].unique()
)

# 根据选择过滤DataFrame - 使用正确的列名 'boro'
df_filtered = df[df['boro'].isin(boroughs) & df['cuisine_description'].isin(cuisines)]

# ------------------- PAGE 1: CITY OVERVIEW -------------------
if page == "City Overview":
    st.title("🏙️ NYC Restaurant Health Overview")
    st.markdown("Explore the spatial distribution and overall health status of NYC's food industry.")

    # KPIs
    col1, col2, col3 = st.columns(3)
    total_records = len(df)
    avg_score = df['score'].mean() if 'score' in df else 0
    critical_rate = (df['critical_flag'] == 'Y').mean() if 'critical_flag' in df else 0
    total_inspections = df_filtered.shape[0]
    a_grade_percentage = (df_filtered[df_filtered['grade'] == 'A'].shape[0] / total_inspections) * 100 if total_inspections > 0 else 0
    col1.metric("Total Inspections", f"{total_inspections:,}")
    col2.metric("Avg. Score", f"{avg_score:.2f}")
    col3.metric("Grade 'A' Restaurants (%)", f"{a_grade_percentage:.1f}%")

    st.markdown("---")

    # Left column: Map (scatter mapbox)
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Restaurant Geographic Distribution")
        if 'latitude' in df and 'longitude' in df:
            sample_df = df.sample(n=min(3000, total_records))
            fig_map = px.scatter_mapbox(
                sample_df, lat="latitude", lon="longitude",
                color="boro" if 'boro' in df else None,
                hover_name="dba" if 'dba' in df else None,
                zoom=10, center={"lat": 40.7128, "lon": -74.0060},
                mapbox_style="carto-positron",
                title="Restaurants by Borough (sampled)"
            )
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Latitude/longitude columns not available. Map cannot be displayed.")

    with col_right:
        st.subheader("Score Trend Over Years")
        if 'year' in df and 'score' in df:
            time_data = df_filtered.set_index('inspection_date').resample('YE')['score'].mean().reset_index()
            time_data['inspection_date'] = time_data['inspection_date'].dt.year # 只显示年份
            fig_time = px.line(time_data, x='inspection_date', y='score', title="Yearly Average Health Score Trend", markers=True)
            fig_time.update_xaxes(title_text='Year')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Year or score column missing.")

    st.markdown("---")

    # Second row: Cuisine ranking & Income vs Score (if available)
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Top 20 Cuisines by Score")
        if 'cuisine_clean' in df and 'score' in df:
            cuisine_perf = df_filtered.groupby('cuisine_clean')['score'].mean().sort_values(ascending=False).head(20).reset_index()
            fig_cuisine = px.bar(cuisine_perf, y='cuisine_clean', x='score', orientation='h',
                                 title="               Best Average Scores", labels={'cuisine_clean': 'Cuisine', 'score': 'Avg. Score'})
            st.plotly_chart(fig_cuisine, use_container_width=True)
        else:
            st.info("Cuisine or score column missing.")

    with col_b:
        st.subheader("Income vs. Health Score")
        if 'median_income' in df and 'score' in df and 'zipcode' in df:
            # Aggregate by zipcode
            zip_agg = df_filtered.groupby('zipcode').agg(
                avg_score=('score', 'mean'),
                median_income=('median_income', 'first')
            ).dropna().reset_index()
            fig_scatter = px.scatter(zip_agg, x='median_income', y='avg_score',
                                     trendline='ols', trendline_color_override='red',
                                     title="Socio‑economic Impact on Health Scores",
                                     labels={'median_income': 'Median Income ($)', 'avg_score': 'Avg. Score'})
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Income, zipcode, or score data not available.")

# ------------------- PAGE 2: VIOLATION DEEP DIVE -------------------
elif page == "Violation Deep Dive":
    st.title("🔍 Violation Patterns by Cuisine & Borough")
    st.markdown("Explore LDA‑derived violation topics and their relationship with cuisine types and locations.")

  
    st.subheader("Violation Topics by Cuisine")
    if 'cuisine_clean' in df and 'violation_topic_name' in df:
        order_series = df_filtered['cuisine_clean'].value_counts().sort_values(ascending=False)
        sorted_cuisines = order_series.index.tolist()
        fig_cuisine_topic = px.bar(df_filtered, x='cuisine_clean', color='violation_topic_name',
                                    title="Cuisine vs. Violation Topic",
                                    labels={'cuisine_clean': 'Cuisine', 'count': 'Number of Violations'},
                                    category_orders={"cuisine_clean": sorted_cuisines},
                                    barmode='stack')
        st.plotly_chart(fig_cuisine_topic, use_container_width=True)
    else:
        st.info("Required columns (cuisine_clean / violation_topic_name) missing.")

    
    st.subheader("Score Trends by Borough")
    if 'year' in df and 'boro' in df and 'score' in df:
        boro_trend = df_filtered.groupby(['year', 'boro'])['score'].mean().reset_index()
        fig_boro = px.line(boro_trend, x='year', y='score', color='boro',
                            title="Average Score Over Years by Borough")
        st.plotly_chart(fig_boro, use_container_width=True)
    else:
        st.info("Year, borough, or score column missing.")

    st.markdown("---")

    # Additional: Violation topic trend over time (optional)
    st.subheader("Violation Topic Evolution Over Time")
    if 'year' in df and 'violation_topic_name' in df:
        topic_trend = df_filtered.groupby(['year', 'violation_topic_name']).size().reset_index(name='count')
        fig_topic_time = px.area(topic_trend, x='year', y='count', color='violation_topic_name',
                                 title="Number of Violations by Topic per Year",
                                 labels={'count': 'Violation Count', 'year': 'Year'})
        st.plotly_chart(fig_topic_time, use_container_width=True)
    else:
        st.info("Cannot show topic evolution – missing year or violation_topic_name.")

    # Optional: raw data viewer
    if st.checkbox("Show filtered violation data"):
        st.dataframe(df[['inspection_date', 'boro', 'cuisine_clean', 'violation_topic_name', 'score']].head(100))

# ------------------- PAGE 3: CLOSURE RISK PREDICTION -------------------
elif page == "Closure Risk Prediction":
<<<<<<< HEAD
    st.title("🔮 Predictive Analytics: Restaurant Closure Risk")
    st.markdown("Using a **LightGBM Classifier** to estimate closure probability.")

    if model is None:
        st.error("Model package not found. Please check if 'restaurant_closure_model2_fixed.pkl' exists in your repository.")
    else:
        # --- 核心修复点：从字典中提取真正的模型和特征名 ---
        # 如果你保存时用了字典打包，这里需要解包
        if isinstance(model, dict):
            actual_model = model.get('model')
            model_features = model.get('features')
        else:
            # 如果加载出来直接就是模型对象
            actual_model = model
            try:
                model_features = actual_model.feature_name()
            except:
                model_features = ['critical_cnt', 'score_last', 'inspection_freq', 'median_income', 
                                  'boro_Manhattan', 'boro_Brooklyn', 'boro_Queens', 'boro_Bronx', 'boro_Staten Island']
                
=======
    st.title("🔮 Restaurant Closure Risk Prediction")
    st.markdown("Using a **Random Forest model** trained on inspection history and socio‑economic data to estimate the risk of closure within the next year.")

    if model is None:
        st.error("❌ Model file `restaurant_closure_model_fixed.pkl` not found. Please place it in the current directory.")
    else:
        # Get expected feature names from the model
        try:
            model_features = model.feature_names_in_
        except AttributeError:
            st.error("Model does not contain `feature_names_in_`. Ensure it was trained with a DataFrame.")
            st.stop()

>>>>>>> parent of 1b5dfae (LGBM)
        with st.form("prediction_form"):
            st.subheader("Enter Restaurant Characteristics")
            col1, col2 = st.columns(2)

            with col1:
                critical_cnt = st.number_input("Recent critical violations", 0, 50, 2)
                score_last = st.number_input("Last inspection score", 0, 150, 12)
                inspection_freq = st.slider("Annual inspection frequency", 0.0, 10.0, 1.5)

            with col2:
                # Borough must match the one‑hot encoded columns in the model
                boro_options = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
                boro_choice = st.selectbox("Borough", boro_options)
                median_income = st.number_input("Census tract median income ($)", 20000, 250000, 75000)

            submit = st.form_submit_button("Run Risk Assessment")

        if submit:
<<<<<<< HEAD
            # 【修复点 2】使用定义的 model_features 创建矩阵
            input_df = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)
            
            mapping = {
=======
            # Create input dataframe with all zeros, then fill known values
            input_df = pd.DataFrame(np.zeros((1, len(model_features))), columns=model_features)

            # Fill numeric features (adjust column names as needed)
            numeric_mapping = {
>>>>>>> parent of 1b5dfae (LGBM)
                'critical_cnt': critical_cnt,
                'score_last': score_last,
                'inspection_freq': inspection_freq,
                'median_income': median_income,
            }
            for col in input_df.columns:
                if col in numeric_mapping:
                    input_df[col] = numeric_mapping[col]
                elif col.startswith('boro_'):
                    # One‑hot encode borough
                    input_df[col] = 1 if col == f'boro_{boro_choice}' else 0
                # Other features (e.g., time‑based) remain 0 – adjust if your model expects them

<<<<<<< HEAD
            # 处理 Boro One-hot
            boro_col = f"boro_{boro_choice}"
            if boro_col in input_df.columns:
                input_df[boro_col] = 1
            
            # 执行预测
            try:
                risk_probability = actual_model.predict_proba(input_df)[0][1]
                
                st.markdown("---")
                st.subheader(f"Results: {risk_probability:.1%} Closure Probability")
                
                # 进度条展示风险程度
                st.progress(risk_probability)
                
                if risk_probability > 0.325:
                    st.error("🚨 **High Risk Profile**")
                else:
                    st.success("✅ **Stable Profile**")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
=======
            # Predict probability of closure (assuming positive class = closure)
            prob = model.predict_proba(input_df)[0, 1]
            st.subheader("Risk Assessment Result")
            st.metric("Predicted Closure Probability", f"{prob:.1%}")
            if prob >= 0.5:
                st.error("⚠️ High risk: The restaurant is likely to close within the next year.")
            else:
                st.success("✅ Low risk: The restaurant appears financially and operationally stable.")

            # Optional: display feature importance for transparency (if model is a random forest)
            if hasattr(model, 'feature_importances_'):
                with st.expander("Model feature importances (top 10)"):
                    importances = pd.DataFrame({
                        'feature': model_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)
                    st.dataframe(importances)
>>>>>>> parent of 1b5dfae (LGBM)

# ------------------- END -------------------