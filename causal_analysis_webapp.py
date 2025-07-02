# ==================== 因果推断分析与Web应用 ====================
# 肥胖风险因果推断分析和交互式Web应用

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

# 因果推断相关库
try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.dataset import synthetic_data
    CAUSAL_ML_AVAILABLE = True
except ImportError:
    CAUSAL_ML_AVAILABLE = False
    st.warning("CausalML未安装，部分因果推断功能将不可用。请运行: pip install causalml")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP未安装，模型解释功能将不可用。请运行: pip install shap")

# ==================== 数据加载和预处理 ====================
@st.cache_data
def load_and_preprocess_data():
    """加载和预处理肥胖数据集"""
    try:
        # 尝试加载训练数据
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')
        
        # 合并数据进行完整分析
        if 'NObeyesdad' not in test_data.columns:
            # 如果测试集没有目标变量，只使用训练集
            data = train_data.copy()
        else:
            data = pd.concat([train_data, test_data], ignore_index=True)
            
    except FileNotFoundError:
        # 如果文件不存在，生成示例数据
        st.warning("未找到数据文件，使用示例数据")
        data = generate_sample_data()
    
    return data

def generate_sample_data(n_samples=1000):
    """生成示例肥胖数据"""
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
    
    # 生成BMI
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    
    # 生成目标变量
    obesity_types = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 
                    'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
    
    # 基于BMI生成目标变量
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

# ==================== 因果推断分析模块 ====================
class CausalAnalysis:
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        self.treatment_effects = {}
        
    def preprocess_for_causal_analysis(self):
        """为因果推断预处理数据"""
        data = self.data.copy()
        
        # 编码分类变量
        le = LabelEncoder()
        categorical_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 
                           'CAEC', 'MTRANS', 'NObeyesdad']
        
        for col in categorical_cols:
            if col in data.columns:
                data[col + '_encoded'] = le.fit_transform(data[col].astype(str))
        
        # 创建BMI特征
        if 'BMI' not in data.columns:
            data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        
        # 创建治疗变量（运动干预）
        data['exercise_treatment'] = (data['FAF'] >= 2).astype(int)  # 运动频率>=2作为治疗
        
        # 创建结果变量（肥胖风险）
        obesity_risk_map = {
            'Insufficient_Weight': 0, 'Normal_Weight': 0, 'Overweight_Level_I': 1,
            'Overweight_Level_II': 1, 'Obesity_Type_I': 2, 'Obesity_Type_II': 2, 'Obesity_Type_III': 2
        }
        data['obesity_risk'] = data['NObeyesdad'].map(obesity_risk_map)
        
        self.processed_data = data
        return data
    
    def propensity_score_matching(self, treatment_col, outcome_col, covariates):
        """倾向性评分匹配"""
        if self.processed_data is None:
            self.preprocess_for_causal_analysis()
        
        data = self.processed_data.copy()
        
        # 计算倾向性评分
        X = data[covariates]
        y = data[treatment_col]
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练倾向性评分模型
        ps_model = LogisticRegression(random_state=42)
        ps_model.fit(X_scaled, y)
        
        # 计算倾向性评分
        ps_scores = ps_model.predict_proba(X_scaled)[:, 1]
        data['propensity_score'] = ps_scores
        
        # 简单的最近邻匹配
        treated = data[data[treatment_col] == 1].copy()
        control = data[data[treatment_col] == 0].copy()
        
        matched_pairs = []
        
        for _, treated_unit in treated.iterrows():
            # 找到倾向性评分最接近的对照单位
            distances = np.abs(control['propensity_score'] - treated_unit['propensity_score'])
            if len(distances) > 0:
                closest_idx = distances.idxmin()
                matched_pairs.append((treated_unit.name, closest_idx))
                # 移除已匹配的对照单位
                control = control.drop(closest_idx)
        
        # 创建匹配后的数据集
        matched_treated_idx = [pair[0] for pair in matched_pairs]
        matched_control_idx = [pair[1] for pair in matched_pairs]
        
        matched_data = pd.concat([
            data.loc[matched_treated_idx],
            data.loc[matched_control_idx]
        ])
        
        # 计算平均治疗效应 (ATE)
        ate = (matched_data[matched_data[treatment_col] == 1][outcome_col].mean() - 
               matched_data[matched_data[treatment_col] == 0][outcome_col].mean())
        
        return ate, matched_data, ps_scores
    
    def instrumental_variable_analysis(self, treatment_col, outcome_col, instrument_col, covariates):
        """工具变量分析"""
        if self.processed_data is None:
            self.preprocess_for_causal_analysis()
        
        data = self.processed_data.copy()
        
        # 第一阶段：工具变量对治疗变量的回归
        X_first = data[covariates + [instrument_col]]
        y_first = data[treatment_col]
        
        first_stage = LogisticRegression()
        first_stage.fit(X_first, y_first)
        
        # 预测治疗变量
        predicted_treatment = first_stage.predict_proba(X_first)[:, 1]
        
        # 第二阶段：预测的治疗变量对结果变量的回归
        X_second = data[covariates].copy()
        X_second['predicted_treatment'] = predicted_treatment
        y_second = data[outcome_col]
        
        second_stage = LogisticRegression()
        second_stage.fit(X_second, y_second)
        
        # 工具变量估计的治疗效应
        iv_effect = second_stage.coef_[0][-1]  # 预测治疗变量的系数
        
        return iv_effect, first_stage, second_stage
    
    def difference_in_differences(self, data_before, data_after, treatment_col, outcome_col):
        """双重差分分析"""
        # 为演示目的，创建面板数据结构
        data_before['period'] = 0
        data_after['period'] = 1
        
        panel_data = pd.concat([data_before, data_after])
        
        # 创建交互项
        panel_data['treatment_period'] = panel_data[treatment_col] * panel_data['period']
        
        # DID回归
        from sklearn.linear_model import LinearRegression
        
        X = panel_data[[treatment_col, 'period', 'treatment_period']]
        y = panel_data[outcome_col]
        
        did_model = LinearRegression()
        did_model.fit(X, y)
        
        # DID估计量是交互项的系数
        did_effect = did_model.coef_[2]
        
        return did_effect, did_model

# ==================== Web应用界面 ====================
def main():
    st.set_page_config(
        page_title="肥胖风险因果推断分析系统",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 肥胖风险因果推断分析系统")
    st.markdown("### 基于因果推断的肥胖风险分析与干预效果评估")
    
    # 侧边栏导航
    st.sidebar.title("📊 分析模块")
    analysis_type = st.sidebar.selectbox(
        "选择分析类型",
        ["数据概览", "因果推断分析", "实时风险评估", "干预效果模拟", "模型解释"]
    )
    
    # 加载数据
    data = load_and_preprocess_data()
    causal_analyzer = CausalAnalysis(data)
    
    if analysis_type == "数据概览":
        show_data_overview(data)
    elif analysis_type == "因果推断分析":
        show_causal_analysis(causal_analyzer)
    elif analysis_type == "实时风险评估":
        show_real_time_assessment(causal_analyzer)
    elif analysis_type == "干预效果模拟":
        show_intervention_simulation(causal_analyzer)
    elif analysis_type == "模型解释":
        show_model_explanation(data)

def show_data_overview(data):
    """显示数据概览"""
    st.header("📈 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总样本数", len(data))
    with col2:
        st.metric("特征数量", len(data.columns))
    with col3:
        obesity_rate = (data['NObeyesdad'].str.contains('Obesity')).mean() * 100
        st.metric("肥胖率", f"{obesity_rate:.1f}%")
    with col4:
        avg_bmi = data['Weight'].mean() / (data['Height'].mean() ** 2)
        st.metric("平均BMI", f"{avg_bmi:.1f}")
    
    # 数据分布可视化
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("肥胖类型分布")
        obesity_counts = data['NObeyesdad'].value_counts()
        fig = px.pie(values=obesity_counts.values, names=obesity_counts.index,
                    title="肥胖类型分布")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("BMI分布")
        if 'BMI' not in data.columns:
            data['BMI'] = data['Weight'] / (data['Height'] ** 2)
        fig = px.histogram(data, x='BMI', nbins=30, title="BMI分布")
        st.plotly_chart(fig, use_container_width=True)
    
    # 相关性热力图
    st.subheader("特征相关性分析")
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = data[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="特征相关性热力图")
        st.plotly_chart(fig, use_container_width=True)

def show_causal_analysis(causal_analyzer):
    """显示因果推断分析"""
    st.header("🔬 因果推断分析")
    
    # 预处理数据
    processed_data = causal_analyzer.preprocess_for_causal_analysis()
    
    st.subheader("1. 倾向性评分匹配 (PSM)")
    
    # 设置分析参数
    col1, col2 = st.columns(2)
    
    with col1:
        treatment_var = st.selectbox(
            "选择治疗变量",
            ['exercise_treatment', 'FAVC_encoded', 'SMOKE_encoded'],
            help="治疗变量代表我们想要分析因果效应的干预措施"
        )
    
    with col2:
        outcome_var = st.selectbox(
            "选择结果变量",
            ['obesity_risk', 'BMI', 'Weight'],
            help="结果变量是我们想要测量治疗效果的目标变量"
        )
    
    # 选择协变量
    available_covariates = ['Age', 'Gender_encoded', 'Height', 'FCVC', 'NCP', 
                           'CH2O', 'family_history_with_overweight_encoded']
    covariates = st.multiselect(
        "选择协变量（混淆因子）",
        available_covariates,
        default=['Age', 'Gender_encoded', 'Height'],
        help="协变量是可能同时影响治疗和结果的变量"
    )
    
    if st.button("🚀 执行倾向性评分匹配"):
        if len(covariates) > 0:
            try:
                ate, matched_data, ps_scores = causal_analyzer.propensity_score_matching(
                    treatment_var, outcome_var, covariates
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.success(f"**平均治疗效应 (ATE): {ate:.4f}**")
                    st.write(f"匹配后样本数: {len(matched_data)}")
                    
                    # 显示治疗效应解释
                    if ate > 0:
                        st.info(f"治疗组的{outcome_var}平均比对照组高{ate:.4f}个单位")
                    else:
                        st.info(f"治疗组的{outcome_var}平均比对照组低{abs(ate):.4f}个单位")
                
                with col2:
                    # 倾向性评分分布
                    fig = go.Figure()
                    
                    treated_ps = processed_data[processed_data[treatment_var] == 1]['propensity_score'] if 'propensity_score' in processed_data.columns else ps_scores[processed_data[treatment_var] == 1]
                    control_ps = processed_data[processed_data[treatment_var] == 0]['propensity_score'] if 'propensity_score' in processed_data.columns else ps_scores[processed_data[treatment_var] == 0]
                    
                    fig.add_trace(go.Histogram(x=treated_ps, name='治疗组', opacity=0.7))
                    fig.add_trace(go.Histogram(x=control_ps, name='对照组', opacity=0.7))
                    
                    fig.update_layout(title='倾向性评分分布', xaxis_title='倾向性评分', yaxis_title='频数')
                    st.plotly_chart(fig, use_container_width=True)
                
                # 平衡性检验
                st.subheader("协变量平衡性检验")
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
                            '协变量': covar,
                            '治疗组均值': treated_mean,
                            '对照组均值': control_mean,
                            '标准化差异': std_diff
                        })
                
                balance_df = pd.DataFrame(balance_results)
                st.dataframe(balance_df)
                
                # 平衡性可视化
                fig = px.bar(balance_df, x='协变量', y='标准化差异',
                           title='匹配后协变量平衡性（标准化差异 < 0.1 表示良好平衡）')
                fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                            annotation_text="平衡阈值 (0.1)")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"分析过程中出现错误: {str(e)}")
        else:
            st.warning("请至少选择一个协变量")
    
    # 工具变量分析
    st.subheader("2. 工具变量分析 (IV)")
    
    instrument_var = st.selectbox(
        "选择工具变量",
        ['family_history_with_overweight_encoded', 'Gender_encoded', 'Age'],
        help="工具变量应该与治疗相关，但只通过治疗影响结果"
    )
    
    if st.button("🔧 执行工具变量分析"):
        try:
            iv_effect, first_stage, second_stage = causal_analyzer.instrumental_variable_analysis(
                treatment_var, outcome_var, instrument_var, covariates
            )
            
            st.success(f"**工具变量估计的治疗效应: {iv_effect:.4f}**")
            
            # 第一阶段结果
            st.write("**第一阶段回归结果（工具变量 → 治疗变量）:**")
            first_stage_score = first_stage.score(processed_data[covariates + [instrument_var]], 
                                                 processed_data[treatment_var])
            st.write(f"第一阶段R²: {first_stage_score:.4f}")
            
        except Exception as e:
            st.error(f"工具变量分析出现错误: {str(e)}")

def show_real_time_assessment(causal_analyzer):
    """显示实时风险评估"""
    st.header("⚡ 实时肥胖风险评估")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("个人信息输入")
        age = st.slider("年龄", 15, 80, 30)
        gender = st.selectbox("性别", ['Male', 'Female'])
        height = st.number_input("身高 (m)", 1.4, 2.2, 1.7, step=0.01)
        weight = st.number_input("体重 (kg)", 40.0, 200.0, 70.0, step=0.1)
        
    with col2:
        st.subheader("生活方式信息")
        favc = st.selectbox("经常食用高热量食物", ['yes', 'no'])
        fcvc = st.slider("蔬菜摄入频率", 1, 3, 2)
        faf = st.slider("运动频率", 0, 3, 1)
        ch2o = st.slider("每日饮水量", 1, 3, 2)
        family_history = st.selectbox("家族肥胖史", ['yes', 'no'])
    
    if st.button("🔍 评估风险"):
        # 计算BMI
        bmi = weight / (height ** 2)
        
        # 创建用户数据
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
        
        # 风险评估
        risk_score = calculate_risk_score(user_data)
        
        # 显示结果
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("BMI", f"{bmi:.1f}")
        with col2:
            st.metric("风险评分", f"{risk_score:.1f}/100")
        with col3:
            risk_level = get_risk_level(risk_score)
            st.metric("风险等级", risk_level)
        
        # 风险仪表盘
        fig = create_risk_gauge(risk_score)
        st.plotly_chart(fig, use_container_width=True)
        
        # 个性化建议
        recommendations = generate_recommendations(user_data, risk_score)
        st.subheader("💡 个性化建议")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

def show_intervention_simulation(causal_analyzer):
    """显示干预效果模拟"""
    st.header("🎯 干预效果模拟")
    
    st.write("模拟不同干预措施对肥胖风险的因果效应")
    
    # 干预参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("干预措施设置")
        exercise_increase = st.slider("运动频率增加", 0, 3, 1)
        diet_improvement = st.slider("饮食质量改善", 0, 2, 1)
        water_increase = st.slider("饮水量增加", 0, 2, 1)
    
    with col2:
        st.subheader("目标人群")
        target_age_range = st.slider("年龄范围", 18, 70, (25, 45))
        target_gender = st.selectbox("目标性别", ['All', 'Male', 'Female'])
        target_bmi_range = st.slider("BMI范围", 15.0, 45.0, (25.0, 35.0))
    
    if st.button("🚀 模拟干预效果"):
        # 模拟干预效果
        simulation_results = simulate_intervention_effects(
            causal_analyzer.data,
            exercise_increase,
            diet_improvement,
            water_increase,
            target_age_range,
            target_gender,
            target_bmi_range
        )
        
        # 显示结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("干预效果预测")
            for intervention, effect in simulation_results['effects'].items():
                st.metric(f"{intervention}效果", f"{effect:.3f}")
        
        with col2:
            st.subheader("受益人群分析")
            st.write(f"目标人群数量: {simulation_results['target_population']}")
            st.write(f"预期受益人数: {simulation_results['expected_beneficiaries']}")
            st.write(f"受益率: {simulation_results['benefit_rate']:.1%}")
        
        # 效果可视化
        fig = create_intervention_visualization(simulation_results)
        st.plotly_chart(fig, use_container_width=True)

def show_model_explanation(data):
    """显示模型解释"""
    st.header("🔍 模型解释与特征重要性")
    
    if not SHAP_AVAILABLE:
        st.warning("SHAP库未安装，无法显示详细的模型解释")
        return
    
    # 训练简单模型用于解释
    processed_data = preprocess_for_explanation(data)
    
    # 只选择数值型特征和编码后的分类特征
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    # 排除目标变量
    feature_cols = [col for col in numeric_cols if col not in ['NObeyesdad', 'obesity_risk']]
    
    X = processed_data[feature_cols]
    y = processed_data['obesity_risk'] if 'obesity_risk' in processed_data.columns else processed_data['NObeyesdad']
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("特征重要性排序")
        fig = px.bar(feature_importance.head(10), 
                    x='importance', y='feature', 
                    orientation='h',
                    title="Top 10 重要特征")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("模型性能")
        st.metric("测试集准确率", f"{accuracy:.3f}")
        
        # 显示分类报告
        st.text("分类报告:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    # SHAP解释 - 使用训练集的一部分进行解释
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:100])  # 使用测试集的前100个样本
        
        st.subheader("SHAP特征重要性")
        
        # 如果是多分类，选择第一个类别的SHAP值
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values
        
        # 创建SHAP摘要图
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values_plot, X_test.iloc[:100], plot_type="bar", show=False)
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"SHAP分析出现错误: {str(e)}")

# ==================== 辅助函数 ====================
def calculate_risk_score(user_data):
    """计算风险评分"""
    score = 0
    
    # BMI评分
    bmi = user_data['BMI']
    if bmi < 18.5:
        score += 10
    elif bmi < 25:
        score += 0
    elif bmi < 30:
        score += 30
    else:
        score += 60
    
    # 生活方式评分
    if user_data['FAVC'] == 'yes':
        score += 15
    
    score += (3 - user_data['FCVC']) * 5  # 蔬菜摄入越少分数越高
    score += (3 - user_data['FAF']) * 10  # 运动越少分数越高
    score += (3 - user_data['CH2O']) * 5  # 饮水越少分数越高
    
    if user_data['family_history_with_overweight'] == 'yes':
        score += 20
    
    return min(score, 100)

def get_risk_level(score):
    """获取风险等级"""
    if score < 30:
        return "低风险"
    elif score < 60:
        return "中等风险"
    else:
        return "高风险"

def create_risk_gauge(risk_score):
    """创建风险仪表盘"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "肥胖风险评分"},
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
    """生成个性化建议"""
    recommendations = []
    
    if user_data['BMI'] > 25:
        recommendations.append("🎯 建议控制体重，目标BMI在18.5-24.9之间")
    
    if user_data['FAF'] < 2:
        recommendations.append("🏃 增加运动频率，建议每周至少150分钟中等强度运动")
    
    if user_data['FCVC'] < 3:
        recommendations.append("🥬 增加蔬菜摄入，建议每天至少5份蔬果")
    
    if user_data['CH2O'] < 2:
        recommendations.append("💧 增加饮水量，建议每天至少8杯水")
    
    if user_data['FAVC'] == 'yes':
        recommendations.append("🚫 减少高热量食物摄入，选择健康的替代品")
    
    if risk_score > 60:
        recommendations.append("⚠️ 建议咨询专业医生，制定个性化的健康管理计划")
    
    return recommendations

def simulate_intervention_effects(data, exercise_increase, diet_improvement, water_increase, 
                                age_range, target_gender, bmi_range):
    """模拟干预效果"""
    # 筛选目标人群
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
    
    # 模拟干预效果（简化计算）
    effects = {
        '运动干预': exercise_increase * 0.1,  # 假设每增加1单位运动频率减少0.1的肥胖风险
        '饮食改善': diet_improvement * 0.08,
        '饮水增加': water_increase * 0.05
    }
    
    total_effect = sum(effects.values())
    
    # 计算受益人群
    target_population = len(target_data)
    expected_beneficiaries = int(target_population * min(total_effect, 0.8))  # 最多80%受益
    benefit_rate = expected_beneficiaries / target_population if target_population > 0 else 0
    
    return {
        'effects': effects,
        'total_effect': total_effect,
        'target_population': target_population,
        'expected_beneficiaries': expected_beneficiaries,
        'benefit_rate': benefit_rate
    }

def create_intervention_visualization(simulation_results):
    """创建干预效果可视化"""
    effects = simulation_results['effects']
    
    fig = go.Figure(data=[
        go.Bar(name='干预效果', x=list(effects.keys()), y=list(effects.values()))
    ])
    
    fig.update_layout(
        title='不同干预措施的预期效果',
        xaxis_title='干预措施',
        yaxis_title='效果大小',
        showlegend=False
    )
    
    return fig

def preprocess_for_explanation(data):
    """为模型解释预处理数据"""
    processed_data = data.copy()
    
    # 编码分类变量
    le = LabelEncoder()
    categorical_cols = ['Gender', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 
                       'CAEC', 'MTRANS']
    
    for col in categorical_cols:
        if col in processed_data.columns:
            processed_data[col + '_encoded'] = le.fit_transform(processed_data[col].astype(str))
    
    # 创建BMI特征
    if 'BMI' not in processed_data.columns:
        processed_data['BMI'] = processed_data['Weight'] / (processed_data['Height'] ** 2)
    
    # 创建肥胖风险分类
    if 'NObeyesdad' in processed_data.columns:
        obesity_risk_map = {
            'Insufficient_Weight': 0, 'Normal_Weight': 0, 'Overweight_Level_I': 1,
            'Overweight_Level_II': 1, 'Obesity_Type_I': 2, 'Obesity_Type_II': 2, 'Obesity_Type_III': 2
        }
        processed_data['obesity_risk'] = processed_data['NObeyesdad'].map(obesity_risk_map)
    
    return processed_data

if __name__ == "__main__":
    main()