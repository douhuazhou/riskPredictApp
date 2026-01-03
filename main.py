# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import tempfile
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from streamlit.components.v1 import html as st_html
import time

# Type definitions
SklearnModel = RandomForestClassifier
SklearnScaler = StandardScaler

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(
        page_title="Clinical Prediction System",
        layout="wide",
        page_icon="🏥"
    )
st.title("🏥 28-day Mortality Risk Prediction of Sepsis Patients")

# 在页面开头添加自定义CSS（放在st.set_page_config之后）
st.markdown("""
    <style>
        /* 隐藏expander的折叠箭头 */
        div[data-testid="stExpander"] > div:first-child > div:first-child > svg {
            display: none;
        }
        /* 调整expander标题样式，接近popover */
        div[data-testid="stExpander"] > div:first-child > div:first-child {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

numeric_features = ['admission_age', 'sofa', 'SII', 'NLR', 'NAR',  'MLR', 'APAR', 'creatinine', 'bun', 'pt']

## scaled background data
background_data = pd.read_csv("background_data.csv", encoding="GBK")
bk_data_with_features = background_data[['admission_age', 'sofa', 'SII', 'NLR', 'NAR',  'MLR', 'APAR', 'creatinine', 'bun', 'pt']]

# dedinition of COMPOSITE INDICATORS（sub features and calculation formula）
COMPOSITE_INDICATORS = {
    "SII": {
        "sub_features": ["Plt", "Neu", "Lym"], # sub feature of SII
        "formula": lambda p, n, l: p * n / l if l != 0 else 0.0,
        "default_values": [440.0, 4.347, 1.323]  # default value of sub feature
    },
    "NLR": {
        "sub_features": ["Neu", "Lym"],
        "formula": lambda n, l: n / l if l != 0 else 0.0,
        "default_values": [4.347, 1.323]
    },
    "NAR": {
        "sub_features": ["Neu", "Alb"],
        "formula": lambda n, a: n / a if a != 0 else 0.0,
        "default_values": [4.347, 28.0]
    },
    "MLR": {
        "sub_features": ["Mono", "Lym"],
        "formula": lambda m, l: m / l if l != 0 else 0.0,
        "default_values": [0.189, 1.323]
    },
    "APAR": {
        "sub_features": ["Alp", "Alb"],
        "formula": lambda alp, alb: alp / alb if alb != 0 else 0.0,
        "default_values": [35, 28]
    }
}

# ---------------------------
# initialize Session State
# ---------------------------
def init_session_state():
    # initialize composite indicators（SII、NLR...）
    if "SII_value" not in st.session_state:
        st.session_state["SII_value"] = 1445.714286  # initialize default value
    if "NLR_value" not in st.session_state:
        st.session_state["NLR_value"] = 3.285714286  # initialize default value
    if "NAR_value" not in st.session_state:
        st.session_state["NAR_value"] = 0.15525  # initialize default value
    if "MLR_value" not in st.session_state:
        st.session_state["MLR_value"] = 0.142857143  # initialize default value
    if "APAR_value" not in st.session_state:
        st.session_state["APAR_value"] = 1.25  # initialize default value
    
    # initialize sub feature value
    for indicator, info in COMPOSITE_INDICATORS.items():
        for sub_feat, default_val in zip(info["sub_features"], info["default_values"]):
            key = f"{indicator}_{sub_feat}_value"
            if key not in st.session_state:
                st.session_state[key] = default_val

# initialize session
init_session_state()

def ensure_reduced_scaler():
    full_scaler_path = "model/scaler.pkl"
    reduced_scaler_path = "model/scaler_reduced.pkl"
    selected_features = numeric_features

    if not os.path.exists(reduced_scaler_path):
        scaler_full = joblib.load(full_scaler_path)
        selected_indices = [list(scaler_full.feature_names_in_).index(f) for f in selected_features]

        scaler_reduced = StandardScaler()
        scaler_reduced.mean_ = scaler_full.mean_[selected_indices]
        scaler_reduced.scale_ = scaler_full.scale_[selected_indices]
        scaler_reduced.var_ = scaler_full.var_[selected_indices]
        scaler_reduced.n_features_in_ = len(selected_features)
        scaler_reduced.feature_names_in_ = np.array(selected_features, dtype=object)

        joblib.dump(scaler_reduced, reduced_scaler_path)
        print("✅ Reduced scaler created and saved.")

@st.cache_resource(show_spinner="Loading prediction model and scaler...")  # loading indication
def load_model_and_scaler():
    """Load model and scaler"""
    try:
        model_dir = "model"
        model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler_reduced.pkl"))
        return model, scaler
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()

def prepare_input_data(input_data: pd.DataFrame, scaler: SklearnScaler):
    # Standardize numeric features
    temp_df = input_data[numeric_features]
    temp_df_scaled = pd.DataFrame(
        scaler.transform(temp_df.loc[:, scaler.feature_names_in_]),
        columns=scaler.feature_names_in_,
        index=temp_df.index
    )
    input_data[numeric_features] = temp_df_scaled
    return input_data

def make_prediction(model: SklearnModel, input_data: pd.DataFrame) -> float:
    """Perform prediction"""
    try:
        return model.predict_proba(input_data)[0, 1]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop() 

def generate_shap_plot(model: SklearnModel, input_data: pd.DataFrame) -> str:
    """Generate optimized SHAP visualization"""
    try:
        if input_data.empty:
            raise ValueError("input data for shap is empty")
        
        # 1. create shap explainer (KernelExplainer for any model）
        model_feature_names = model.feature_names_in_
        explainer = shap.KernelExplainer(lambda X: model.predict_proba(pd.DataFrame(X, columns=model_feature_names)), bk_data_with_features.head(20))
        ## inout data target value 
        shap_values = explainer.shap_values(input_data)
        
        # first sample
        sample_idx = 0
        sample_shap = shap_values[0][:, 1]
        sample_data = input_data.iloc[sample_idx]
        
        # generate force plot
        fig = shap.plots.force(
            base_value=explainer.expected_value[1],
            shap_values=sample_shap,
            features=sample_data,
            feature_names=input_data.columns.tolist(),
            matplotlib=False,
            plot_cmap="coolwarm",
            text_rotation=15, # decrease  label rotation
            figsize=(12, 6)
        )


        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
            shap.save_html(tmp.name, fig)

            with open(tmp.name, "r", encoding="utf-8") as f:
                html_content = f.read()

            custom_style = """
            <style>
                #container { width: 100% !important; height: 550px !important; padding: 15px !important; }
                .feature-name { font-size: 11px !important; transform: translateY(4px) rotate(15deg) !important; opacity: 0.9 !important; }
                .value { font-size: 10px !important; transform: translateY(-2px) !important; opacity: 0.8 !important; }
                .base-value, .output-value { font-size: 12px !important; font-weight: 600 !important; transform: translate(5px, 15px) !important; }
                .arrow { stroke-width: 1.2 !important; opacity: 0.7 !important; }
                .color-scale { transform: translateY(10px) !important; }
                .hover-info, .x-axis-label { display: none !important; }
                .force-plot .labels > * { margin: 2px 0 !important; }
            </style>
            """
            html_content = html_content.replace('</head>', f'{custom_style}</head>')
            return html_content
    except Exception as e:
        st.error(f"SHAP plot generation failed: {str(e)}")
        st.write("{shap_values}")
        st.stop()
    finally:
        if tmp.name and os.path.exists(tmp.name):
            try:
                os.remove(tmp.name)
            except OSError:
                pass

def calculate_composite_indicator(indicator_name: str) -> None:
    # calculate composite indicator，update Session State composite value
    if indicator_name not in COMPOSITE_INDICATORS:
        return
    
    info = COMPOSITE_INDICATORS[indicator_name]
    sub_values = []
    
    # collect sub features
    for sub_feat in info["sub_features"]:
        key = f"{indicator_name}_{sub_feat}_value"
        sub_values.append(st.session_state[key])
    
    # calculate
    composite_value = info["formula"](*sub_values)

    # time.sleep(0.05)

    # update composite value
    st.session_state[f"{indicator_name}_value"] = round(composite_value, 2)

def main():
    # Ensure reduced scaler exists before loading
    ensure_reduced_scaler()

    # load model and data scaler
    model, scaler = load_model_and_scaler()

    with st.container():
        st.subheader("Enter Patient Data")
        # first row, five column
        col1_row1, col2_row1, col3_row1, col4_row1, col5_row1 = st.columns(5)
        with col1_row1:
            age = st.number_input("Age", value=50.087269, format="%.6f")
        with col2_row1:
            sofa = st.number_input("SOFA", value=13.0, format="%.2f")
        with col3_row1:
            Creatinine = st.number_input("Creatinine", value=1.3, format="%.2f")
        with col4_row1:
            Bun = st.number_input("Bun", value=17.0, format="%.2f")
        with col5_row1:
            Pt= st.number_input("Pt", value=12.5, format="%.2f")


        # second row, five column
        col1_row2, col2_row2, col3_row2, col4_row2, col5_row2 = st.columns(5)
        with col1_row2:
            # SII：only read input + pop for sub features
            with st.popover("📝 Edit SII (Click to calculate)", use_container_width=True):
                st.markdown("### SII Sub-indicators")
                # display SII Sub-indicators
                sii_info = COMPOSITE_INDICATORS["SII"]
                for i, (sub_feat, default_val) in enumerate(zip(sii_info["sub_features"], sii_info["default_values"])):
                    key = f"SII_{sub_feat}_value"
                    st.number_input(
                        f"{sub_feat}",
                        step=0.01,
                        key=key
                        # on_change=calculate_composite_indicator,  # calculate automatically
                        # args=("SII",)
                    )
                # button for calculte SII
                if st.button("Calculate SII", key="calc_sii"):
                    calculate_composite_indicator("SII")
                    st.success(f"SII calculated successfully: {st.session_state['SII_value']}")
    
            # display SII res-only read
            SII = st.text_input(
                "SII",
                value=str(st.session_state["SII_value"]),
                disabled=True,
                help="Click the 'Edit SII' popover to calculate via sub-indicators"
            )
        with col2_row2:
            with st.popover("📝 Edit NLR (Click to calculate)", use_container_width=True):
                st.markdown("### NLR Sub-indicators")
                nlr_info = COMPOSITE_INDICATORS["NLR"]
                for i, (sub_feat, default_val) in enumerate(zip(nlr_info["sub_features"], nlr_info["default_values"])):
                    key = f"NLR_{sub_feat}_value"
                    st.number_input(
                        f"{sub_feat}",
                        step=0.01,
                        key=key,
                        on_change=calculate_composite_indicator,
                        args=("NLR",)
                    )
                if st.button("Calculate NLR", key="calc_nlr"):
                    calculate_composite_indicator("NLR")
                    st.success(f"NLR calculated successfully: {st.session_state['NLR_value']}")
            
            
            NLR = st.text_input(
                "NLR",
                value=str(st.session_state["NLR_value"]),
                disabled=True,
                help="Click the 'Edit NLR' popover to calculate via sub-indicators"
            )
        with col3_row2:
            with st.popover("📝 Edit NAR (Click to calculate)", use_container_width=True):
                st.markdown("### NAR Sub-indicators")
                nar_info = COMPOSITE_INDICATORS["NAR"]
                for i, (sub_feat, default_val) in enumerate(zip(nar_info["sub_features"], nar_info["default_values"])):
                    key = f"NAR_{sub_feat}_value"
                    st.number_input(
                        f"{sub_feat}",
                        step=0.01,
                        key=key,
                        on_change=calculate_composite_indicator,
                        args=("NAR",)
                    )
                if st.button("Calculate NAR", key="calc_nar"):
                    calculate_composite_indicator("NAR")
                    st.success(f"NAR calculated successfully: {st.session_state['NAR_value']}")
            
            NAR = st.text_input(
                "NAR",
                value=str(st.session_state["NAR_value"]),
                disabled=True,
                help="Click the 'Edit NAR' popover to calculate via sub-indicators"
            )
        with col4_row2:
            with st.popover("📝 Edit MLR (Click to calculate)", use_container_width=True):
                st.markdown("### MLR Sub-indicators")
                mlr_info = COMPOSITE_INDICATORS["MLR"]
                for i, (sub_feat, default_val) in enumerate(zip(mlr_info["sub_features"], mlr_info["default_values"])):
                    key = f"MLR_{sub_feat}_value"
                    st.number_input(
                        f"{sub_feat}",
                        step=0.01,
                        key=key,
                        on_change=calculate_composite_indicator,
                        args=("MLR",)
                    )
                if st.button("Calculate MLR", key="calc_mlr"):
                    calculate_composite_indicator("MLR")
                    st.success(f"MLR calculated successfully: {st.session_state['MLR_value']}")
            
            MLR = st.text_input(
                "MLR",
                value=str(st.session_state["MLR_value"]),
                disabled=True,
                help="Click the 'Edit MLR' popover to calculate via sub-indicators"
            )
        with col5_row2:
            with st.popover("📝 Edit APAR (Click to calculate)", use_container_width=True):
                st.markdown("### APAR Sub-indicators")
                apar_info = COMPOSITE_INDICATORS["APAR"]
                for i, (sub_feat, default_val) in enumerate(zip(apar_info["sub_features"], apar_info["default_values"])):
                    key = f"APAR_{sub_feat}_value"
                    st.number_input(
                        f"{sub_feat}",
                        step=1,
                        key=key,
                        on_change=calculate_composite_indicator,
                        args=("APAR",)
                    )
                if st.button("Calculate APAR", key="calc_apar"):
                    calculate_composite_indicator("APAR")
                    st.success(f"APAR calculated successfully: {st.session_state['APAR_value']}")
            
            APAR = st.text_input(
                "APAR",
                value=str(st.session_state["APAR_value"]),
                disabled=True,
                help="Click the 'Edit APAR' popover to calculate via sub-indicators"
            )

    inputs = {
            'admission_age': float(age),
            'sofa': float(sofa),
            'SII': float(SII),
            'NLR': float(NLR),
            'NAR': float(NAR),
            'MLR': float(MLR),
            'APAR': float(APAR),
            'creatinine': float(Creatinine),
            'bun': float(Bun),
            'pt': float(Pt)
        }

    input_df = prepare_input_data(pd.DataFrame([inputs]), scaler)

    if st.button("Start Risk Assessment", type="primary"):
        with st.status("Analyzing...", expanded=True) as status:
            try:
                if input_df.isnull().any().any():
                    raise ValueError("Input data contains invalid values")

                risk = make_prediction(model, input_df)

                status.update(label="Analysis complete", state="complete")


                # create one row
                col1 = st.columns(1)[0]

                with col1:
                    st.subheader("Risk Assessment Result")
                    st.metric("Probability of Mortality", f"{risk * 100:.1f}%")

                    st.subheader("Key Influencing Factors")
                    html_content = generate_shap_plot(model, input_df)
                    st_html(html_content, height=600, scrolling=False)

            except Exception as e:
                print(str(e))
                status.update(label="Analysis failed", state="error")
                st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()