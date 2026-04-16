import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

# 设置页面配置
st.set_page_config(page_title="NYC Smart Restaurant Analytics", layout="wide")

# --- 数据加载 (建议使用缓存加速) ---
@st.cache_data
def load_data():
    # 这里加载您处理后的 df_with_violation_topic.csv
    df = pd.read_csv("df_with_violation_topic.csv")
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    return df

@st.cache_resource
def load_model():
    # 加载您在 random forest.ipynb 中保存的模型
    return joblib.load('restaurant_closure_model_fixed.pkl')

df = load_data()
model = load_model()

# --- 侧边栏导航 ---
st.sidebar.title("项目导航")
page = st.sidebar.radio("选择分析板块", ["城市概览", "违规行为分析", "倒闭风险预测"])

# --- 板块 1: 城市概览 ---
if page == "城市概览":
    st.title("🏙️ 纽约市餐厅卫生安全概览")
    
    # 指标卡片
    col1, col2, col3 = st.columns(3)
    col1.metric("总记录数", f"{len(df):,}")
    col2.metric("平均分 (Score)", f"{df['score'].mean():.2f}")
    col3.metric("高风险违规占比", f"{(df['critical_flag'] == 'Y').mean():.1%}")

    # 地图展示
    st.subheader("餐厅空间分布")
    fig_map = px.scatter_mapbox(df.head(1000), lat="latitude", lon="longitude", color="boro", 
                               hover_name="dba", zoom=10, mapbox_style="carto-positron")
    st.plotly_chart(fig_map, use_container_width=True)

# --- 板块 2: 违规行为分析 (灵感二) ---
elif page == "违规行为分析":
    st.title("🔍 违规特征与社区/菜系关系")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("不同菜系的违规主题分布")
        # 基于您 Visualization.ipynb 中的逻辑
        cuisine_list = df['cuisine_clean'].unique().tolist()
        selected_cuisine = st.multiselect("选择菜系", cuisine_list, default=cuisine_list[:5])
        
        mask = df['cuisine_clean'].isin(selected_cuisine)
        fig_cuisine = px.bar(df[mask], x="cuisine_clean", color="violation_topic_name", 
                             title="菜系 vs 违规主题", barmode="stack")
        st.plotly_chart(fig_cuisine)

    with col_b:
        st.subheader("行政区评分走势")
        df_trend = df.groupby(['year', 'boro'])['score'].mean().reset_index()
        fig_trend = px.line(df_trend, x="year", y="score", color="boro", title="历年评分趋势")
        st.plotly_chart(fig_trend)

# --- 板块 3: 倒闭风险预测 (灵感一) ---
elif page == "倒闭风险预测":
    st.title("🔮 餐厅生存风险预测")
    st.write("基于随机森林模型预测餐厅在未来一年内的经营风险。")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            critical_cnt = st.number_input("近期严重违规次数 (critical_cnt)", 0, 50, 2)
            score_last = st.number_input("最近一次检查分数", 0, 150, 12)
            inspection_freq = st.slider("年均检查频率", 0.0, 10.0, 1.5)
        
        with col2:
            boro_choice = st.selectbox("所在行政区", df['boro'].unique())
            median_income = st.number_input("社区收入中位数 (USD)", 20000, 250000, 75000)
        
        submit = st.form_submit_button("开始评估")

    if submit:
        # 这里的特征构建需与 random forest.ipynb 中的训练特征严格一致
        # 示例：假设模型需要 [critical_cnt, score_last, inspection_freq, median_income]
        input_data = np.array([[critical_cnt, score_last, inspection_freq, median_income]])
        
        prediction_prob = model.predict_proba(input_data)[0][1]
        
        st.subheader("评估结果")
        if prediction_prob > 0.6:
            st.error(f"高风险！倒闭概率预计为: {prediction_prob:.1%}")
            st.warning("建议：加强卫生管理，减少严重违规项以降低闭店风险。")
        else:
            st.success(f"经营状态稳健。倒闭概率预计为: {prediction_prob:.1%}")

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("ECON7990 Group Project - Smart City Data Analytics")