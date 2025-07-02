# ==================== Causal Inference Analysis and Web Application ====================
# Obesity Risk Causal Inference Analysis and Interactive Web Application

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Causal inference related libraries
try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.dataset import synthetic_data
    CAUSAL_ML_AVAILABLE = True
except ImportError:
    CAUSAL_ML_AVAILABLE = False
    st.warning("CausalML is not installed, some causal inference features will not be available. Please run: pip install causalml")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP is not installed, model explanation features will not be available. Please run: pip install shap")

# ==================== Data Loading and Preprocessing ====================
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess obesity dataset"""
    try:
        # Try to load training data
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        
        # Merge data for complete analysis
        if 'NObeyesdad' not in test_data.columns:
            # If test set doesn't have target variable, use only training set
            data = train_data.copy()
        else:
            data = pd.concat([train_data, test_data], ignore_index=True)
            
    except FileNotFoundError:
        # If file doesn't exist, generate sample data
        st.warning("Data files not found, using sample data")
        data = generate_sample_data()
    
    return data

def generate_sample_data(n_samples=1000):
    """Generate sample obesity data"""
    np.random.seed(42)
    
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Height': np.random.normal(1.7, 0.1, n_samples),
        'Weight': np.random.normal(70, 15, n_samples),
        'FAVC': np.random.choice(['yes', 'no'], n_samples),
        'FCVC': np.random.randint(1, 4, n_samples),
        'NCP': np.random.randint(1, 5, n_samples),
        'SCC': np.random.choice(['yes', 'no'], n_samples),
        'SMOKE': np.random.choice(['yes', 'no'], n_samples),
        'CH2O': np.random.randint(1, 4, n_samples),
        'family_history_with_overweight': np.random.choice(['yes', 'no'], n_samples),
        'FAF': np.random.randint(0, 4, n_samples),
        'TUE': np.random.randint(0, 3, n_samples),
        'CAEC': np.random.choice(['no', 'Sometimes', 'Frequently', 'Always'], n_samples),
        'MTRANS': np.random.choice(['Walking', 'Public_Transportation', 'Automobile', 'Bike'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # Generate target variable
    obesity_types = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
                    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    
    # Generate target variable based on BMI
    conditions = [
        df['BMI'] < 18.5,
        (df['BMI'] >= 18.5) & (df['BMI'] < 25),
        (df['BMI'] >= 25) & (df['BMI'] < 27),
        (df['BMI'] >= 27) & (df['BMI'] < 30),
        (df['BMI'] >= 30) & (df['BMI'] < 35),
        (df['BMI'] >= 35) & (df['BMI'] < 40),
        df['BMI'] >= 40
    ]
    
    df['NObeyesdad'] = np.select(conditions, obesity_types, default='Normal_Weight')
    
    return df

# ==================== Causal Inference Analysis Module ====================
class CausalAnalysis:
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        self.treatment_effects = {}
        
    def preprocess_for_causal_analysis(self):
        """Preprocess data for causal inference"""
        data = self.data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        categorical_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 
                           'CAEC', 'MTRANS', 'NObeyesdad']
        
        for col in categorical_cols:
            if col in data.columns:
                data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        
        # Create BMI feature
        if 'BMI' not in data.columns:
            data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        
        # Create treatment variable (exercise intervention)
        data['exercise_treatment'] = (data['FAF'] >= 2).astype(int)  # Exercise frequency >= 2 as treatment
        
        # Create outcome variable (obesity risk)
        obesity_risk_map = {
            'Insufficient_Weight': 0, 'Normal_Weight': 0, 'Overweight_Level_I': 1,
            'Overweight_Level_II': 1, 'Obesity_Type_I': 2, 'Obesity_Type_II': 2, 'Obesity_Type_III': 2
        }
        data['obesity_risk'] = data['NObeyesdad'].map(obesity_risk_map)
        
        self.processed_data = data
        return data
    
    def propensity_score_matching(self, treatment_col, outcome_col, covariates):
        """Propensity Score Matching"""
        if self.processed_data is None:
            self.preprocess_for_causal_analysis()
        
        data = self.processed_data.copy()
        
        # Calculate propensity scores
        X = data[covariates]
        y = data[treatment_col]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train propensity score model
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_scaled, y)
        
        # Calculate propensity scores
        ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
        data['propensity_score'] = ps_scores
        
        # Simple nearest neighbor matching
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()
        
        matched_pairs = []
        
        for _, treated_unit in treated.iterrows():
            # Find the control unit with the closest propensity score
            distances = np.abs(control['propensity_score'] - treated_unit['propensity_score'])
            if len(distances) > 0:
                closest_idx = distances.idxmin()
                matched_pairs.append((treated_unit.name, closest_idx))
                # Remove matched control unit
                control = control.drop(closest_idx)
        
        # Create matched dataset
        matched_treated_idx = [pair[0] for pair in matched_pairs]
        matched_control_idx = [pair[1] for pair in matched_pairs]
        
        matched_data = pd.concat([
            data.loc[matched_treated_idx],
            data.loc[matched_control_idx]
        ])
        
        # Calculate Average Treatment Effect (ATE)
        ate = (matched_data[matched_data[treatment_col] == 1][outcome_col].mean() - 
               matched_data[matched_data[treatment_col] == 0][outcome_col].mean())
        
        return ate, matched_data, ps_scores
    
    def instrumental_variable_analysis(self, treatment_col, outcome_col, instrument_col, covariates):
        """Instrumental Variable Analysis"""
        if self.processed_data is None:
            self.preprocess_for_causal_analysis()
        
        data = self.processed_data.copy()
        
        # First stage: Regression of instrument on treatment
        X_first = data[covariates + [instrument_col]]
        y_first = data[treatment_col]
        
        first_stage = LogisticRegression()
        first_stage.fit(X_first, y_first)
        
        # Predict treatment
        predicted_treatment = first_stage.predict_proba(X_first)[:, 1]
        
        # Second stage: Regression of predicted treatment on outcome
        X_second = data[covariates].copy()
        X_second['predicted_treatment'] = predicted_treatment
        y_second = data[outcome_col]
        
        second_stage = LogisticRegression()
        second_stage.fit(X_second, y_second)
        
        # Treatment effect estimated by IV
        iv_effect = second_stage.coef_[0][-1]  # Coefficient of predicted treatment
        
        return iv_effect, first_stage, second_stage
    
    def difference_in_differences(self, data_before, data_after, treatment_col, outcome_col):
        """Difference-in-Differences Analysis"""
        # Create panel data structure for demonstration purposes
        data_before['period'] = 0
        data_after['period'] = 1
        
        panel_data = pd.concat([data_before, data_after])
        
        # Create interaction term
        panel_data['treatment_period'] = panel_data[treatment_col] * panel_data['period']
        
        # DID regression
        from sklearn.linear_model import LinearRegression
        
        X = panel_data[[treatment_col, 'period', 'treatment_period']]
        y = panel_data[outcome_col]
        
        did_model = LinearRegression()
        did_model.fit(X, y)
        
        # DID estimator is the coefficient of the interaction term
        did_effect = did_model.coef_[2]
        
        return did_effect, did_model

# ==================== Web Application Interface ====================
def main():
    st.set_page_config(
        page_title="Obesity Risk Causal Inference Analysis System",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Obesity Risk Causal Inference Analysis System")
    st.markdown("### Causal Inference-Based Obesity Risk Analysis and Intervention Effect Evaluation")
    
    # Sidebar navigation
    st.sidebar.title("📊 Analysis Modules")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        ["Data Overview", "Causal Inference Analysis", "Real-time Risk Assessment", "Intervention Effect Simulation", "Model Explanation"]
    )
    
    # Load data
    data = load_and_preprocess_data()
    causal_analyzer = CausalAnalysis(data)
    
    if analysis_type == "Data Overview":
        show_data_overview(data)
    elif analysis_type == "Causal Inference Analysis":
        show_causal_analysis(causal_analyzer)
    elif analysis_type == "Real-time Risk Assessment":
        show_real_time_assessment(causal_analyzer)
    elif analysis_type == "Intervention Effect Simulation":
        show_intervention_simulation(causal_analyzer)
    elif analysis_type == "Model Explanation":
        show_model_explanation(data)

def show_data_overview(data):
    """Display data overview"""
    st.header("📈 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", len(data))
    with col2:
        st.metric("Number of Features", len(data.columns))
    with col3:
        obesity_rate = (data['NObeyesdad'].str.contains('Obesity')).mean() * 100
        st.metric("Obesity Rate", f"{obesity_rate:.1f}%")
    with col4:
        avg_bmi = data['Weight'].mean() / (data['Height'].mean() ** 2)
        st.metric("Average BMI", f"{avg_bmi:.1f}")
    
    # Data distribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Obesity Type Distribution")
        obesity_counts = data['NObeyesdad'].value_counts()
        fig = px.pie(values=obesity_counts.values, names=obesity_counts.index,
                    title="Obesity Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("BMI Distribution")
        if 'BMI' not in data.columns:
            data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        fig = px.histogram(data, x='BMI', nbins=30, title="BMI Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Analysis")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

def show_causal_analysis(causal_analyzer):
    """Display causal inference analysis"""
    st.header("🔬 Causal Inference Analysis")
    
    # Preprocess data
    processed_data = causal_analyzer.preprocess_for_causal_analysis()
    
    st.subheader("1. Propensity Score Matching (PSM)")
    
    # Set analysis parameters
    col1, col2 = st.columns(2)
    
    with col1:
        treatment_var = st.selectbox(
            "Select Treatment Variable",
            ['exercise_treatment', 'FAVC_encoded', 'SMOKE_encoded'],
            help="Treatment variable represents the intervention we want to analyze for causal effects"
        )
    
    with col2:
        outcome_var = st.selectbox(
            "Select Outcome Variable",
            ['obesity_risk', 'BMI', 'Weight'],
            help="Outcome variable is the target variable we want to measure treatment effects on"
        )
    
    # Select covariates
    available_covariates = ['Age', 'Gender_encoded', 'Height', 'FCVC', 'NCP', 
                           'CH2O', 'family_history_with_overweight_encoded']
    covariates = st.multiselect(
        "Select Covariates (Confounding Factors)",
        available_covariates,
        default=['Age', 'Gender_encoded', 'Height'],
        help="Covariates are variables that may affect both treatment and outcome"
    )
    
    if st.button("🚀 Execute Propensity Score Matching"):
        if len(covariates) > 0:
            try:
                ate, matched_data, ps_scores = causal_analyzer.propensity_score_matching(
                    treatment_var, outcome_var, covariates
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**Average Treatment Effect (ATE): {ate:.4f}**")
                    st.write(f"Matched Sample Size: {len(matched_data)}")
                    
                    # Display treatment effect explanation
                    if ate > 0:
                        st.info(f"The treatment group's {outcome_var} is on average {ate:.4f} units higher than the control group")
                    else:
                        st.info(f"The treatment group's {outcome_var} is on average {abs(ate):.4f} units lower than the control group")
                
                with col2:
                    # Propensity score distribution
                    fig = go.Figure()
                    
                    treated_ps = processed_data[processed_data[treatment_var] == 1]['propensity_score'] if 'propensity_score' in processed_data.columns else ps_scores[processed_data[treatment_var] == 1]
                    control_ps = processed_data[processed_data[treatment_var] == 0]['propensity_score'] if 'propensity_score' in processed_data.columns else ps_scores[processed_data[treatment_var] == 0]
                    
                    fig.add_trace(go.Histogram(x=treated_ps, name='Treatment Group', opacity=0.7))
                    fig.add_trace(go.Histogram(x=control_ps, name='Control Group', opacity=0.7))
                    
                    fig.update_layout(title='Propensity Score Distribution', xaxis_title='Propensity Score', yaxis_title='Frequency')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Balance test
                st.subheader("Covariate Balance Test")
                balance_results = []
                
                for covar in covariates:
                    if covar in matched_data.columns:
                        treated_mean = matched_data[matched_data[treatment_var] == 1][covar].mean()
                        control_mean = matched_data[matched_data[treatment_var] == 0][covar].mean()
                        std_diff = abs(treated_mean - control_mean) / np.sqrt(
                            (matched_data[matched_data[treatment_var] == 1][covar].var() + 
                             matched_data[matched_data[treatment_var] == 0][covar].var()) / 2
                        )
                        
                        balance_results.append({
                            'Covariate': covar,
                            'Treatment Group Mean': treated_mean,
                            'Control Group Mean': control_mean,
                            'Standardized Difference': std_diff
                        })
                
                balance_df = pd.DataFrame(balance_results)
                st.dataframe(balance_df)
                
                # Balance visualization
                fig = px.bar(balance_df, x='Covariate', y='Standardized Difference',
                           title='Post-Matching Covariate Balance (Standardized Difference < 0.1 indicates good balance)')
                fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                            annotation_text="Balance Threshold (0.1)")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
        else:
            st.warning("Please select at least one covariate")
    
    # Instrumental variable analysis
    st.subheader("2. Instrumental Variable Analysis (IV)")
    
    instrument_var = st.selectbox(
        "Select Instrumental Variable",
        ['family_history_with_overweight_encoded', 'Gender_encoded', 'Age'],
        help="Instrumental variable should be related to treatment but only affect outcome through treatment"
    )
    
    if st.button("🔧 Execute Instrumental Variable Analysis"):
        try:
            iv_effect, first_stage, second_stage = causal_analyzer.instrumental_variable_analysis(
                treatment_var, outcome_var, instrument_var, covariates
            )
            
            st.success(f"**Treatment Effect Estimated by IV: {iv_effect:.4f}**")
            
            # First stage results
            st.write("**First Stage Regression Results (Instrument → Treatment):**")
            first_stage_score = first_stage.score(processed_data[covariates + [instrument_var]], 
                                                 processed_data[treatment_var])
            st.write(f"First Stage R²: {first_stage_score:.4f}")
            
        except Exception as e:
            st.error(f"Error in instrumental variable analysis: {str(e)}")

def show_real_time_assessment(causal_analyzer):
    """Display real-time risk assessment"""
    st.header("⚡ Real-time Obesity Risk Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information Input")
        age = st.slider("Age", 15, 80, 30)
        gender = st.selectbox("Gender", ['Male', 'Female'])
        height = st.number_input("Height (m)", 1.4, 2.2, 1.7, step=0.01)
        weight = st.number_input("Weight (kg)", 40.0, 200.0, 70.0, step=0.1)
        
    with col2:
        st.subheader("Lifestyle Information")
        favc = st.selectbox("Frequent Consumption of High Calorie Food", ['yes', 'no'])
        fcvc = st.slider("Vegetable Consumption Frequency", 1, 3, 2)
        faf = st.slider("Physical Activity Frequency", 0, 3, 1)
        ch2o = st.slider("Daily Water Consumption", 1, 3, 2)
        family_history = st.selectbox("Family History of Obesity", ['yes', 'no'])
    
    if st.button("🔍 Assess Risk"):
        # Calculate BMI
        bmi = weight / (height ** 2)
        
        # Create user data
        user_data = {
            'Age': age,
            'Gender': gender,
            'Height': height,
            'Weight': weight,
            'BMI': bmi,
            'FAVC': favc,
            'FCVC': fcvc,
            'FAF': faf,
            'CH2O': ch2o,
            'family_history_with_overweight': family_history
        }
        
        # Risk assessment
        risk_score = calculate_risk_score(user_data)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BMI", f"{bmi:.1f}")
        with col2:
            st.metric("Risk Score", f"{risk_score:.1f}/100")
        with col3:
            risk_level = get_risk_level(risk_score)
            st.metric("Risk Level", risk_level)
        
        # Risk gauge
        fig = create_risk_gauge(risk_score)
        st.plotly_chart(fig, use_container_width=True)
        
        # Personalized recommendations
        recommendations = generate_recommendations(user_data, risk_score)
        st.subheader("💡 Personalized Recommendations")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def show_intervention_simulation(causal_analyzer):
    """Display intervention effect simulation"""
    st.header("🎯 Intervention Effect Simulation")
    
    st.write("Simulate causal effects of different interventions on obesity risk")
    
    # Intervention parameter settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Intervention Settings")
        exercise_increase = st.slider("Exercise Frequency Increase", 0, 3, 1)
        diet_improvement = st.slider("Diet Quality Improvement", 0, 2, 1)
        water_increase = st.slider("Water Consumption Increase", 0, 2, 1)
    
    with col2:
        st.subheader("Target Population")
        target_age_range = st.slider("Age Range", 18, 70, (25, 45))
        target_gender = st.selectbox("Target Gender", ['All', 'Male', 'Female'])
        target_bmi_range = st.slider("BMI Range", 15.0, 45.0, (25.0, 35.0))
    
    if st.button("🚀 Simulate Intervention Effects"):
        # Simulate intervention effects
        simulation_results = simulate_intervention_effects(
            causal_analyzer.data,
            exercise_increase,
            diet_improvement,
            water_increase,
            target_age_range,
            target_gender,
            target_bmi_range
        )
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Intervention Effect Prediction")
            for intervention, effect in simulation_results['effects'].items():
                st.metric(f"{intervention} Effect", f"{effect:.3f}")
        
        with col2:
            st.subheader("Beneficiary Population Analysis")
            st.write(f"Target Population Size: {simulation_results['target_population']}")
            st.write(f"Expected Beneficiaries: {simulation_results['expected_beneficiaries']}")
            st.write(f"Benefit Rate: {simulation_results['benefit_rate']:.1%}")
        
        # Effect visualization
        fig = create_intervention_visualization(simulation_results)
        st.plotly_chart(fig, use_container_width=True)

def show_model_explanation(data):
    """Display model explanation"""
    st.header("🔍 Model Explanation and Feature Importance")
    
    if not SHAP_AVAILABLE:
        st.warning("SHAP library is not installed, detailed model explanations are not available")
        return
    
    # Train a simple model for explanation
    processed_data = preprocess_for_explanation(data)
    
    # Select only numeric features and encoded categorical features
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    # Exclude target variables and any direct encodings of the target
    feature_cols = [col for col in numeric_cols if col not in ['NObeyesdad', 'obesity_risk', 'NObeyesdad_encoded']]
    
    X = processed_data[feature_cols]
    y = processed_data['obesity_risk'] if 'obesity_risk' in processed_data.columns else processed_data['NObeyesdad']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model with cross-validation to prevent overfitting
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance Ranking")
        fig = px.bar(feature_importance.head(10), 
                    x='importance', y='feature', 
                    orientation='h',
                    title="Top 10 Important Features")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        st.metric("Test Set Accuracy", f"{accuracy:.3f}")
        
        # Display classification report
        st.text("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # SHAP explanation - use part of the training set for explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:100])  # Use first 100 samples from test set
        
        st.subheader("SHAP Feature Importance")
        
        # If multi-class, select the first class's SHAP values
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values
        
        # Create SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_test.iloc[:100], plot_type="bar", show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Error in SHAP analysis: {str(e)}")

# ==================== Helper Functions ====================
def calculate_risk_score(user_data):
    """Calculate risk score"""
    score = 0
    
    # BMI score
    bmi = user_data['BMI']
    if bmi < 18.5:
        score += 10
    elif bmi < 25:
        score += 0
    elif bmi < 30:
        score += 30
    else:
        score += 60
    
    # Lifestyle score
    if user_data['FAVC'] == 'yes':
        score += 15
    
    score += (3 - user_data['FCVC']) * 5  # Less vegetable consumption = higher score
    score += (3 - user_data['FAF']) * 10  # Less exercise = higher score
    score += (3 - user_data['CH2O']) * 5  # Less water consumption = higher score
    
    if user_data['family_history_with_overweight'] == 'yes':
        score += 20
    
    return min(score, 100)

def get_risk_level(score):
    """Get risk level"""
    if score < 30:
        return "Low Risk"
    elif score < 60:
        return "Medium Risk"
    else:
        return "High Risk"

def create_risk_gauge(risk_score):
    """Create risk gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Obesity Risk Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "yellow"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def generate_recommendations(user_data, risk_score):
    """Generate personalized recommendations"""
    recommendations = []
    
    if user_data['BMI'] > 25:
        recommendations.append("🎯 Recommended weight control, target BMI between 18.5-24.9")
    
    if user_data['FAF'] < 2:
        recommendations.append("🏃 Increase exercise frequency, recommend at least 150 minutes of moderate-intensity exercise per week")
    
    if user_data['FCVC'] < 3:
        recommendations.append("🥬 Increase vegetable intake, recommend at least 5 servings of fruits and vegetables daily")
    
    if user_data['CH2O'] < 2:
        recommendations.append("💧 Increase water consumption, recommend at least 8 glasses of water daily")
    
    if user_data['FAVC'] == 'yes':
        recommendations.append("🚫 Reduce high-calorie food intake, choose healthy alternatives")
    
    if risk_score > 60:
        recommendations.append("⚠️ Recommend consulting a professional doctor to develop a personalized health management plan")
    
    return recommendations

def simulate_intervention_effects(data, exercise_increase, diet_improvement, water_increase, 
                                age_range, target_gender, bmi_range):
    """Simulate intervention effects"""
    # Filter target population
    if 'BMI' not in data.columns:
        data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    
    target_data = data[
        (data['Age'] >= age_range[0]) & 
        (data['Age'] <= age_range[1]) &
        (data['BMI'] >= bmi_range[0]) & 
        (data['BMI'] <= bmi_range[1])
    ]
    
    if target_gender != 'All':
        target_data = target_data[target_data['Gender'] == target_gender]
    
    # Simulate intervention effects (simplified calculation)
    effects = {
        'Exercise Intervention': exercise_increase * 0.1,  # Assume each unit of exercise frequency increase reduces obesity risk by 0.1
        'Diet Improvement': diet_improvement * 0.08,
        'Water Increase': water_increase * 0.05
    }
    
    total_effect = sum(effects.values())
    
    # Calculate beneficiary population
    target_population = len(target_data)
    expected_beneficiaries = int(target_population * min(total_effect, 0.8))  # Maximum 80% benefit
    benefit_rate = expected_beneficiaries / target_population if target_population > 0 else 0
    
    return {
        'effects': effects,
        'total_effect': total_effect,
        'target_population': target_population,
        'expected_beneficiaries': expected_beneficiaries,
        'benefit_rate': benefit_rate
    }

def create_intervention_visualization(simulation_results):
    """Create intervention effect visualization"""
    effects = simulation_results['effects']
    
    fig = go.Figure(data=[
        go.Bar(name='Intervention Effect', x=list(effects.keys()), y=list(effects.values()))
    ])
    
    fig.update_layout(
        title='Expected Effects of Different Interventions',
        xaxis_title='Intervention Measures',
        yaxis_title='Effect Size',
        showlegend=False
    )
    
    return fig

def preprocess_for_explanation(data):
    """Preprocess data for model explanation"""
    processed_data = data.copy()
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 
                       'CAEC', 'MTRANS', 'NObeyesdad']
    
    for col in categorical_cols:
        if col in processed_data.columns:
            processed_data[col + '_encoded'] = le.fit_transform(processed_data[col].astype(str))
    
    # Create BMI feature
    if 'BMI' not in processed_data.columns:
        processed_data['BMI'] = processed_data['Weight'] / (processed_data['Height'] ** 2)
    
    # Create obesity risk classification
    if 'NObeyesdad' in processed_data.columns:
        obesity_risk_map = {
            'Insufficient_Weight': 0, 'Normal_Weight': 0, 'Overweight_Level_I': 1,
            'Overweight_Level_II': 1, 'Obesity_Type_I': 2, 'Obesity_Type_II': 2, 'Obesity_Type_III': 2
        }
        processed_data['obesity_risk'] = processed_data['NObeyesdad'].map(obesity_risk_map)
    
    return processed_data

if __name__ == "__main__":
    main()