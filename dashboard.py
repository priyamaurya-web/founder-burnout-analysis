import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

# Page Configuration
st.set_page_config(page_title="Founder Burnout Analyzer", layout="wide")
st.title("🚨 Startup Founder Burnout Analyzer 2026")
st.markdown("### Understanding & Predicting Founder Burnout")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/startup_founder_burnout_2026.csv")

df = load_data()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", 
    ["Overview", "Key Insights", "Exploratory Analysis", "Burnout Predictor", "Recommendations"])

# ==================== PAGE 1: OVERVIEW ====================
if page == "Overview":
    st.header("Project Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Founders", f"{len(df):,}")
    with col2:
        burnout_rate = (df['Founder_Burnout_Flag'].mean() * 100).round(1)
        st.metric("Burnout Rate", f"{burnout_rate}%")
    with col3:
        severe = (df['Burnout_Level'] == 'Severe').mean() * 100
        st.metric("Severe Burnout", f"{severe:.1f}%")
    
    st.subheader("What Drives Founder Burnout?")
    st.write("""
    This dashboard explores how work habits, stress, and startup conditions affect founder mental health.
    """)

# ==================== PAGE 2: KEY INSIGHTS ====================
elif page == "Key Insights":
    st.header("🔑 Key Insights")
    st.markdown(open("reports/key_insights.md").read())

# ==================== PAGE 3: EXPLORATORY ANALYSIS ====================
elif page == "Exploratory Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Burnout by Founder Type")
        fig = px.box(df, x="Founder_Type", y="Burnout_Score", color="Burnout_Level")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Work Hours vs Burnout")
        fig2 = px.scatter(df, x="Weekly_Work_Hours", y="Burnout_Score", 
                         color="Burnout_Level", hover_data=["Founder_Type"])
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Correlation Heatmap")
    key_cols = ['Weekly_Work_Hours', 'Sleep_Hours', 'Exercise_Days_Per_Week', 
                'Investor_Pressure_Score', 'Burnout_Score']
    fig3 = px.imshow(df[key_cols].corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig3, use_container_width=True)

# ==================== PAGE 4: BURNOUT PREDICTOR ====================
elif page == "Burnout Predictor":
    st.header("🔮 Burnout Risk Predictor")
    
    # Load Model
    model = joblib.load("models/burnout_predictor_model.pkl")
    
    st.subheader("Enter Founder Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        work_hours = st.slider("Weekly Work Hours", 20, 100, 65)
        sleep_hours = st.slider("Sleep Hours per Night", 3.0, 9.0, 5.5, 0.1)
        exercise = st.slider("Exercise Days per Week", 0, 7, 2)
        investor_pressure = st.slider("Investor Pressure (1-10)", 1, 10, 6)
    
    with col2:
        age = st.slider("Founder Age", 22, 55, 32)
        experience = st.slider("Experience (Years)", 0, 15, 3)
        team_size = st.slider("Team Size", 1, 500, 12)
        pmf_score = st.slider("Product-Market Fit Score", 1, 10, 6)
    
    # Prediction
    if st.button("Predict Burnout Risk", type="primary"):
        input_data = pd.DataFrame({
            'Weekly_Work_Hours': [work_hours],
            'Sleep_Hours': [sleep_hours],
            'Exercise_Days_Per_Week': [exercise],
            'Investor_Pressure_Score': [investor_pressure],
            'Cofounder_Conflict_Score': [5],      # default
            'Decision_Fatigue_Score': [6],        # default
            'Stress_Score': [5],                  # default
            'Work_Intensity': [work_hours / sleep_hours],
            'Total_Stress': [(investor_pressure + 5 + 6 + 5)/4],
            'Founder_Age': [age],
            'Founder_Experience_Years': [experience],
            'Team_Size': [team_size],
            'Startup_Age_Months': [24],           # default
            'Product_Market_Fit_Score': [pmf_score],
            'Runway_Months_Remaining': [12]       # default
        })
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        if prediction == 1:
            st.error(f"⚠️ HIGH RISK of Burnout | Probability: {probability:.1%}")
        else:
            st.success(f"✅ Low Risk | Probability: {probability:.1%}")

# ==================== PAGE 5: RECOMMENDATIONS ====================
elif page == "Recommendations":
    st.header("💡 Recommendations")
    st.markdown("""
    ### For Founders
    - Sleep at least **6.5 hours** daily
    - Keep weekly work hours under **60**
    - Exercise **3+ days** per week
    
    ### For Investors
    - Track founder workload during check-ins
    - Support mental health resources
    
    ### Red Flags
    - Working >75 hours/week
    - Sleeping <5 hours
    - High investor pressure + solo founder
    """)

st.sidebar.markdown("---")
st.sidebar.info("Portfolio Project by [Your Name]")
