"""
急性缺血性卒中机械取栓患者90天预后预测计算器
基于Streamlit和SHAP的网页应用
版本：1.3 (瀑布图优化版)
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

# ==================== 应用初始化 ====================
st.set_page_config(page_title="卒中预后预测计算器", layout="wide")
st.title("脑卒中机械取栓患者发病90天预后情况预测")
st.markdown("""
此工具基于训练的随机森林模型，根据患者的临床特征预测其发病后90天的预后情况。
**免责声明**：本工具旨在辅助临床治疗，不能替代专业医疗判断。
""")

# ==================== 特征配置字典 ====================
FEATURE_CONFIG = {
    'Age': ('年龄 (岁)', {'type': 'number', 'min_value': 18, 'max_value': 120, 'value': 65, 'step': 1}),
    'P': ('脉搏/心率 (次/分)', {'type': 'number', 'min_value': 30, 'max_value': 200, 'value': 85, 'step': 1}),
    'NIHSS': ('NIHSS评分', {'type': 'number', 'min_value': 0, 'max_value': 42, 'value': 10, 'step': 1}),
    'PNT': ('穿刺到再通时间 (分钟)', {'type': 'number', 'min_value': 0, 'max_value': 500, 'value': 220, 'step': 10}),
    'NEUT': ('中性粒细胞计数 (10^9/L)', {'type': 'number', 'min_value': 0.0, 'max_value': 50.0, 'value': 7.0, 'step': 0.1}),
    'D-dimer': ('D-二聚体 (mg/L)', {'type': 'number', 'min_value': 0.0, 'max_value': 20.0, 'value': 0.5, 'step': 0.1}),
    'NLR': ('中性粒细胞-淋巴细胞比值 (NLR)', {'type': 'number', 'min_value': 0.0, 'max_value': 50.0, 'value': 3.0, 'step': 0.1}),
    'Barthel Index Score': ('Barthel指数', {'type': 'number', 'min_value': 0, 'max_value': 100, 'value': 60, 'step': 5}),
    'NRS2002 Score': ('NRS2002营养风险评分', {'type': 'number', 'min_value': 0, 'max_value': 7, 'value': 3, 'step': 1}),
    'Dysphagia': ('入院时吞咽困难', {'type': 'select', 'options': ['无', '有'], 'index': 0}),
}

# ==================== 加载模型与数据 ====================
@st.cache_resource
def load_model_and_explainer():
    """缓存加载模型和SHAP解释器"""
    try:
        model_data = joblib.load('./model.joblib')
        model = model_data['model']
        
        # 提取特征名
        if 'feature_names' in model_data:
            model_feature_names = model_data['feature_names']
        elif 'features' in model_data:
            model_feature_names = model_data['features']
        else:
            model_feature_names = list(FEATURE_CONFIG.keys())
        
        # 验证特征配置
        missing_features = [f for f in model_feature_names if f not in FEATURE_CONFIG]
        if missing_features:
            st.error(f"❌ 特征配置缺失：{missing_features}")
            st.stop()
        
        # 加载数据用于SHAP背景
        df = pd.read_excel('./data.xlsx')
        X_background = df[model_feature_names]
        
        # 创建SHAP解释器
        sample_size = min(50, len(X_background))
        background_data = X_background.sample(sample_size, random_state=42)
        explainer = shap.TreeExplainer(model, data=background_data)
        
        return model, model_feature_names, explainer, model_data
        
    except Exception as e:
        st.error(f"❌ 加载失败: {str(e)}")
        raise

# 尝试加载模型
try:
    model, model_feature_names, shap_explainer, model_metadata = load_model_and_explainer()
    st.sidebar.success("✅ 模型加载成功！")
    st.sidebar.info(f"已加载 {len(model_feature_names)} 个特征")
except Exception as e:
    st.sidebar.error(f"❌ 加载失败: {e}")
    st.stop()

# ==================== 侧边栏输入界面 ====================
st.sidebar.header("🔬 输入患者临床特征")
user_inputs = {}

for feature_name in model_feature_names:
    if feature_name in FEATURE_CONFIG:
        label, config = FEATURE_CONFIG[feature_name]
        
        if config['type'] == 'number':
            user_input = st.sidebar.number_input(
                label=label,
                min_value=config.get('min_value', 0),
                max_value=config.get('max_value', 100),
                value=config.get('value', 0),
                step=config.get('step', 1),
                key=f"input_{feature_name}"
            )
            user_inputs[feature_name] = user_input
            
        elif config['type'] == 'select' and feature_name == 'Dysphagia':
            display_value = st.sidebar.selectbox(
                label=label,
                options=config['options'],
                index=config['index'],
                key=f"select_{feature_name}"
            )
            user_inputs[feature_name] = 1 if display_value == '有' else 0

# ==================== 预测执行 ====================
if st.sidebar.button("🚀 开始预测", type="primary", use_container_width=True):
    with st.spinner('模型计算中...'):
        
        # 1. 准备输入数据
        input_df = pd.DataFrame([user_inputs], columns=model_feature_names)
        
        # 2. 获取预测概率
        proba = model.predict_proba(input_df)[0]
        good_prob = proba[0] * 100
        poor_prob = proba[1] * 100
        
        # 3. 显示预测结果
        st.header("📊 预测结果")
        col1, col2 = st.columns(2)
        with col1:
            progress_value = max(0.0, min(good_prob / 100.0, 1.0))
            st.metric(
                label="**良好预后概率 (mRS 0-2)**",
                value=f"{good_prob:.1f}%",
                delta=f"{good_prob-50:.1f}%" if good_prob >= 50 else f"{good_prob-50:.1f}%"
            )
            st.progress(progress_value)
        with col2:
            progress_value = max(0.0, min(poor_prob / 100.0, 1.0))
            st.metric(
                label="**不良预后概率 (mRS 3-6)**",
                value=f"{poor_prob:.1f}%",
                delta=f"{poor_prob-50:.1f}%" if poor_prob >= 50 else f"{poor_prob-50:.1f}%",
                delta_color="inverse"
            )
            st.progress(progress_value)
        
        # 4. 临床解读
        st.subheader("🧭 临床解读")
        if good_prob >= 70:
            st.success(f"**高可能性良好预后** ({good_prob:.1f}%) - 该患者有较高概率获得良好功能恢复。")
        elif good_prob >= 40:
            st.warning(f"**中等可能性良好预后** ({good_prob:.1f}%) - 需密切监测与积极干预。")
        else:
            st.error(f"**高风险不良预后** ({poor_prob:.1f}%) - 建议采取强化临床管理策略。")
        
        # ==================== SHAP解释部分 ====================
        st.header("🔍 模型解释分析")
        
        # 计算SHAP值
        try:
            shap_values = shap_explainer(input_df)
            
            # 提取基准值
            if isinstance(shap_explainer.expected_value, (list, np.ndarray)):
                base_value = float(shap_explainer.expected_value[1])
            else:
                base_value = float(shap_explainer.expected_value)
            
            # 提取当前样本的SHAP值
            shap_vals = None
            
            if hasattr(shap_values, 'values'):
                if len(shap_values.values.shape) == 3:
                    shap_vals = shap_values.values[0, :, 1]
                else:
                    shap_vals = shap_values.values[0, :]
            elif isinstance(shap_values, list) and len(shap_values) == 2:
                shap_vals = shap_values[1][0, :]
            else:
                shap_vals = np.array(shap_values).flatten()
            
            shap_vals = np.array(shap_vals).flatten()
            final_prediction = base_value + shap_vals.sum()
            
        except Exception as e:
            st.error(f"❌ 计算SHAP值时出错: {str(e)}")
            shap_vals = np.zeros(len(model_feature_names))
            base_value = 0.5
            final_prediction = 0.5
        
        # 5. 创建SHAP瀑布图（已移除所有标题和坐标轴标签）
        st.subheader("📈 特征贡献瀑布图")
        st.markdown("""
        此图展示了各特征如何影响预测结果。从**基准风险**开始，每个特征的贡献（正值增加风险，负值降低风险）依次累加，得到**最终预测值**。
        """)
        
        try:
            # 创建SHAP Explanation对象
            explanation = shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=input_df.iloc[0].values,
                feature_names=model_feature_names
            )
            
            # 绘制瀑布图
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.waterfall_plot(explanation, max_display=len(model_feature_names), show=False)
            
            # 移除所有坐标轴标签，只保留最简洁的图表
            ax.set_xlabel('')  # 移除X轴标签
            ax.set_ylabel('')  # 移除Y轴标签
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"❌ 生成瀑布图时出错: {str(e)}")
            st.info("尝试显示简化的特征贡献表格...")
        
        # 瀑布图解读
        with st.expander("💡 如何解读瀑布图？", expanded=False):
            st.markdown(f"""
            **图表元素解读：**
            1.  **E[f(X)] = {base_value:.3f}** - 基准风险，代表患者群体的平均风险水平
            2.  **每行一个特征** - 显示该特征对预测的具体贡献
            3.  **条形方向**：
                - 🔴 **红色（向右）**：增加不良预后风险
                - 🔵 **蓝色（向左）**：降低不良预后风险
            4.  **条形长度**：贡献的绝对值大小
            5.  **f(x) = {final_prediction:.3f}** - 本次预测的最终逻辑值
            
            **临床意义**：红色条的特征是风险因素，蓝色条的是保护因素。
            """)
        
        # 6. 特征贡献度表格
        st.subheader("📋 特征贡献度明细")
        
        try:
            # 准备表格数据
            table_data = []
            for i, feat_name in enumerate(model_feature_names):
                feat_label = FEATURE_CONFIG[feat_name][0]
                contribution = shap_vals[i]
                abs_contrib = abs(contribution)
                
                table_data.append({
                    '临床特征': feat_label,
                    '贡献值': contribution,
                    '绝对值': abs_contrib,
                    '方向': '🔴 增加风险' if contribution > 0 else '🔵 降低风险'
                })
            
            # 按绝对值排序
            contrib_df = pd.DataFrame(table_data)
            contrib_df = contrib_df.sort_values('绝对值', ascending=False).reset_index(drop=True)
            
            # 显示表格
            st.dataframe(
                contrib_df.style.format({'贡献值': '{:+.4f}', '绝对值': '{:.4f}'}),
                use_container_width=True,
                hide_index=True
            )
            
        except Exception as e:
            st.error(f"❌ 生成贡献度表格时出错: {str(e)}")
        
        # 7. 输入数据回顾
        with st.expander("📝 查看本次输入的详细数据", expanded=False):
            try:
                review_data = []
                for feat_name in model_feature_names:
                    feat_label = FEATURE_CONFIG[feat_name][0]
                    raw_value = user_inputs[feat_name]
                    
                    if feat_name == 'Dysphagia':
                        disp_value = '有' if raw_value == 1 else '无'
                    else:
                        disp_value = raw_value
                    
                    review_data.append({
                        '临床特征': feat_label,
                        '输入值': disp_value
                    })
                
                review_df = pd.DataFrame(review_data)
                st.dataframe(review_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ 显示输入数据时出错: {str(e)}")

# ==================== 页脚信息 ====================
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ 使用说明")
st.sidebar.markdown("""
1. 在左侧输入患者临床特征
2. 点击 **🚀 开始预测** 按钮
3. 查看右侧预测结果和解释
""")

st.sidebar.markdown("---")
model_name = model_metadata.get('model_name', 'Random Forest')
training_date = model_metadata.get('training_date', '未知日期')
st.sidebar.caption(f"**模型信息**：{model_name} | 训练日期：{training_date}")
st.sidebar.caption("开发框架：Streamlit + SHAP")

# ==================== 应用说明 ====================
with st.expander("📖 关于此计算器", expanded=False):
    st.markdown("""
    ### 模型背景
    - **算法**：随机森林分类器
    - **目标变量**：90天改良Rankin量表评分 (mRS)
    - **分类**：良好预后 (mRS 0-2) vs 不良预后 (mRS 3-6)
    
    ### 技术架构
    - **前端界面**：Streamlit
    - **模型解释**：SHAP
    - **部署平台**：Streamlit Community Cloud
    
    ### 注意事项
    1. 本工具适用于**研究目的**，临床决策需结合医生专业判断
    2. 模型在训练数据范围外可能表现不佳
    3. 定期验证和更新模型是必要的
    """)