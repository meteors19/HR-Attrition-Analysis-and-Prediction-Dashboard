# ============================================================
# HR ATTRITION PREDICTION - STREAMLIT APP
# Abhishek Kumar | 2026
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="HR Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- LOAD MODEL ----
@st.cache_resource
def load_model():
    with open('model/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, rf_model, scaler, feature_cols

model, rf_model, scaler, feature_cols = load_model()

# ---- LOAD DATA ----
@st.cache_data
def load_data():
    df = pd.read_csv("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    return df

df = load_data()

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1F4E79, #2E75B6);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background-color: #C00000;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .risk-low {
        background-color: #375623;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown('<div class="main-header">👥 HR Attrition Analysis & Prediction Dashboard</div>',
            unsafe_allow_html=True)
st.markdown("---")

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔮 Predict Attrition", "📈 Feature Insights"])

# ============================================================
# TAB 1: DASHBOARD
# ============================================================
with tab1:
    st.subheader("Company-Wide Attrition Overview")

    # KPI Metrics
    total = len(df)
    left = df[df['Attrition'] == 'Yes'].shape[0]
    stayed = total - left
    rate = round(left / total * 100, 1)
    avg_age_left = round(df[df['Attrition'] == 'Yes']['Age'].mean(), 1)
    avg_tenure_left = round(df[df['Attrition'] == 'Yes']['YearsAtCompany'].mean(), 1)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Employees", f"{total:,}")
    col2.metric("Employees Left", f"{left}")
    col3.metric("Attrition Rate", f"{rate}%", delta=f"-{rate}% target: <10%", delta_color="inverse")
    col4.metric("Retained", f"{stayed:,}")
    col5.metric("Avg Age (Left)", f"{avg_age_left} yrs")
    col6.metric("Avg Tenure (Left)", f"{avg_tenure_left} yrs")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        # Department Attrition
        dept_data = df.groupby('Department')['Attrition'].apply(
            lambda x: round((x == 'Yes').sum() / len(x) * 100, 1)
        ).reset_index()
        dept_data.columns = ['Department', 'Attrition Rate (%)']
        fig1 = px.bar(dept_data.sort_values('Attrition Rate (%)', ascending=False),
                      x='Department', y='Attrition Rate (%)',
                      title='Attrition Rate by Department',
                      color='Attrition Rate (%)',
                      color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        # Overtime Attrition
        ot_data = df.groupby('OverTime')['Attrition'].apply(
            lambda x: round((x == 'Yes').sum() / len(x) * 100, 1)
        ).reset_index()
        ot_data.columns = ['OverTime', 'Attrition Rate (%)']
        fig2 = px.bar(ot_data, x='OverTime', y='Attrition Rate (%)',
                      title='Attrition Rate: Overtime vs No Overtime',
                      color='OverTime',
                      color_discrete_map={'Yes': '#C00000', 'No': '#375623'})
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        # Job Satisfaction
        sat_data = df.groupby('JobSatisfaction')['Attrition'].apply(
            lambda x: round((x == 'Yes').sum() / len(x) * 100, 1)
        ).reset_index()
        sat_data.columns = ['Job Satisfaction', 'Attrition Rate (%)']
        sat_data['Job Satisfaction'] = sat_data['Job Satisfaction'].map(
            {1: '1-Low', 2: '2-Medium', 3: '3-High', 4: '4-Very High'}
        )
        fig3 = px.bar(sat_data, x='Job Satisfaction', y='Attrition Rate (%)',
                      title='Attrition by Job Satisfaction',
                      color='Attrition Rate (%)',
                      color_continuous_scale='RdYlGn_r')
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        # Age distribution
        fig4 = px.histogram(df, x='Age', color='Attrition',
                            title='Age Distribution by Attrition Status',
                            barmode='overlay', opacity=0.7,
                            color_discrete_map={'Yes': '#C00000', 'No': '#2E75B6'})
        st.plotly_chart(fig4, use_container_width=True)

# ============================================================
# TAB 2: PREDICTION
# ============================================================
with tab2:
    st.subheader("🔮 Employee Attrition Risk Predictor")
    st.info("Fill in the employee details below to predict their attrition risk.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        age = st.slider("Age", 18, 60, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        education = st.selectbox("Education Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Below College", 2:"College",
                                                          3:"Bachelor", 4:"Master", 5:"Doctor"}[x])
        distance = st.slider("Distance From Home (km)", 1, 29, 5)

    with col2:
        st.markdown("**Job Details**")
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Job Role", df['JobRole'].unique())
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
        job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4],
                                         format_func=lambda x: {1:"Low", 2:"Medium",
                                                                  3:"High", 4:"Very High"}[x])
        overtime = st.selectbox("Overtime", ["Yes", "No"])

    with col3:
        st.markdown("**Compensation & Experience**")
        monthly_income = st.number_input("Monthly Income (₹)", 1000, 200000, 50000, step=1000)
        years_at_company = st.slider("Years at Company", 0, 40, 3)
        years_in_role = st.slider("Years in Current Role", 0, 18, 2)
        work_life_balance = st.selectbox("Work-Life Balance", [1, 2, 3, 4],
                                          format_func=lambda x: {1:"Bad", 2:"Good",
                                                                   3:"Better", 4:"Best"}[x])
        environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4],
                                                  format_func=lambda x: {1:"Low", 2:"Medium",
                                                                           3:"High", 4:"Very High"}[x])

    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        # Build input — must match training feature order
        input_dict = {col: 0 for col in feature_cols}

        # Map categorical values
        gender_map = {"Male": 1, "Female": 0}
        dept_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
        marital_map = {"Single": 2, "Married": 1, "Divorced": 0}
        overtime_map = {"Yes": 1, "No": 0}
        jobrole_map = {role: i for i, role in enumerate(sorted(df['JobRole'].unique()))}

        # Update values (simplified mapping — adjust to match your actual encoder output)
        updates = {
            'Age': age, 'DistanceFromHome': distance,
            'Education': education, 'EnvironmentSatisfaction': environment_satisfaction,
            'Gender': gender_map.get(gender, 0),
            'JobLevel': job_level, 'JobSatisfaction': job_satisfaction,
            'MaritalStatus': marital_map.get(marital_status, 0),
            'MonthlyIncome': monthly_income, 'OverTime': overtime_map.get(overtime, 0),
            'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
            'YearsInCurrentRole': years_in_role,
            'Department': dept_map.get(department, 0),
            'JobRole': jobrole_map.get(job_role, 0),
        }

        for k, v in updates.items():
            if k in input_dict:
                input_dict[k] = v

        input_df = pd.DataFrame([input_dict])
        input_df = input_df[feature_cols]

        prob = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        risk_pct = round(prob * 100, 1)

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        col_r1, col_r2, col_r3 = st.columns([1, 2, 1])
        with col_r2:
            if prediction == 1:
                st.markdown(f'<div class="risk-high">⚠️ HIGH ATTRITION RISK — {risk_pct}%</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">✅ LOW ATTRITION RISK — {risk_pct}%</div>',
                            unsafe_allow_html=True)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            title={'text': "Attrition Risk (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#C00000" if risk_pct > 50 else "#375623"},
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'value': 50}
            }
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendations
        st.subheader("💡 Recommended Actions")
        if overtime == "Yes":
            st.warning("⚠️ Employee works overtime — consider workload redistribution")
        if job_satisfaction <= 2:
            st.warning("⚠️ Low job satisfaction — schedule a career development discussion")
        if work_life_balance <= 2:
            st.warning("⚠️ Poor work-life balance — consider flexible working arrangements")
        if years_at_company <= 2:
            st.info("ℹ️ Early tenure employee — ensure proper onboarding and mentorship")
        if monthly_income < 30000:
            st.warning("⚠️ Below average compensation — review salary benchmarks")

# ============================================================
# TAB 3: FEATURE INSIGHTS
# ============================================================
with tab3:
    st.subheader("📈 What Drives Attrition?")

    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(15)

    fig_imp = px.bar(importances.sort_values('Importance'),
                     x='Importance', y='Feature',
                     orientation='h',
                     title='Top 15 Features Driving Attrition (Random Forest)',
                     color='Importance',
                     color_continuous_scale='Blues')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Deep Dive: Income vs Attrition")
    fig_box = px.box(df, x='Attrition', y='MonthlyIncome',
                     color='Attrition',
                     title='Monthly Income Distribution by Attrition',
                     color_discrete_map={'Yes': '#C00000', 'No': '#2E75B6'})
    st.plotly_chart(fig_box, use_container_width=True)

# ---- FOOTER ----
st.markdown("---")
st.markdown("**HR Attrition Dashboard** | Built by Abhishek Kumar | IBM HR Dataset | 2026")
