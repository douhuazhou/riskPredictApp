# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import os
import tempfile
from typing import Tuple, Any
from sklearn.preprocessing import StandardScaler
from streamlit.components.v1 import html as st_html

# Type definitions
SklearnModel = Any  # 兼容 RandomForest / CatBoost 等多种模型
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

# 自定义 CSS
st.markdown("""
    <style>
        div[data-testid="stExpander"] > div:first-child > div:first-child > svg {
            display: none;
        }
        div[data-testid="stExpander"] > div:first-child > div:first-child {
            font-weight: 600;
            font-size: 16px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --------- 11 features (加入 PLR) ---------
numeric_features = [
    'admission_age', 'sofa', 'SII', 'NLR', 'PLR',
    'NAR', 'MLR', 'APAR', 'creatinine', 'bun', 'pt'
]

# scaled background data
background_data = pd.read_csv("background_data.csv", encoding="GBK")
bk_data_with_features = background_data[numeric_features]


# --------- COMPOSITE INDICATORS (新增 PLR) ---------
COMPOSITE_INDICATORS = {
    "SII": {
        "sub_features": ["Plt", "Neu", "Lym"],
        "formula": lambda p, n, l: p * n / l if l != 0 else 0.0,
        "default_values": [440.0, 4.347, 1.323]
    },
    "NLR": {
        "sub_features": ["Neu", "Lym"],
        "formula": lambda n, l: n / l if l != 0 else 0.0,
        "default_values": [4.347, 1.323]
    },
    "PLR": {
        "sub_features": ["Plt", "Lym"],
        "formula": lambda p, l: p / l if l != 0 else 0.0,
        "default_values": [440.0, 1.323]
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
    # 6 个复合指标的默认值
    defaults = {
        "SII_value": 1445.714286,
        "NLR_value": 3.285714286,
        "PLR_value": 332.6076,   # 440 / 1.323
        "NAR_value": 0.15525,
        "MLR_value": 0.142857143,
        "APAR_value": 1.25,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # 子特征
    for indicator, info in COMPOSITE_INDICATORS.items():
        for sub_feat, default_val in zip(info["sub_features"], info["default_values"]):
            key = f"{indicator}_{sub_feat}_value"
            if key not in st.session_state:
                st.session_state[key] = default_val


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
        print("Reduced scaler created and saved.")


@st.cache_resource(show_spinner="Loading prediction model and scaler...")
def load_model_and_scaler():
    """Load model and scaler"""
    try:
        model_dir = "model"
        model_file = "cb_model.pkl"
        model = joblib.load(os.path.join(model_dir, model_file))
        scaler = joblib.load(os.path.join(model_dir, "scaler_reduced.pkl"))

        # 校验模型特征集与 UI 一致
        mf = getattr(model, "feature_names_in_", None) or getattr(model, "feature_names_", None)
        if mf is not None:
            mf = list(mf)
            if set(mf) != set(numeric_features):
                missing = set(numeric_features) - set(mf)
                extra = set(mf) - set(numeric_features)
                st.error(
                    f"Model / UI feature mismatch (model={model_file}).\n"
                    f"Model features ({len(mf)}): {mf}\n"
                    f"UI features ({len(numeric_features)}): {numeric_features}\n"
                    f"Missing in model: {missing} | Extra in model: {extra}\n"
                )
                st.stop()

        # 校验 scaler_reduced 的特征集
        sf = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None
        if sf is not None and set(sf) != set(numeric_features):
            st.error(
                f"Scaler feature mismatch.\n"
                f"Scaler features: {sf}\n"
                f"UI features: {numeric_features}\n"
                f"▶ 请删除 model/scaler_reduced.pkl 重新生成。"
            )
            st.stop()

        return model, scaler
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
        st.stop()


def prepare_input_data(input_data: pd.DataFrame, scaler: SklearnScaler):
    temp_df = input_data[numeric_features]
    temp_df_scaled = pd.DataFrame(
        scaler.transform(temp_df.loc[:, scaler.feature_names_in_]),
        columns=scaler.feature_names_in_,
        index=temp_df.index
    )
    input_data[numeric_features] = temp_df_scaled
    return input_data


def make_prediction(model: SklearnModel, input_data: pd.DataFrame) -> float:
    try:
        # 若模型有 feature_names_in_ / feature_names_，按其顺序排列
        feat_order = getattr(model, "feature_names_in_", None)
        if feat_order is None:
            feat_order = getattr(model, "feature_names_", None)
        if feat_order is not None:
            input_data = input_data[list(feat_order)]
        return model.predict_proba(input_data)[0, 1]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        st.stop()


def generate_shap_plot(model: SklearnModel, input_data: pd.DataFrame) -> str:
    """Generate optimized SHAP visualization"""
    try:
        if input_data.empty:
            raise ValueError("input data for shap is empty")

        # 优先使用 model.feature_names_in_ / feature_names_，否则用当前输入列
        model_feature_names = getattr(model, "feature_names_in_", None)
        if model_feature_names is None:
            model_feature_names = getattr(model, "feature_names_", None)
        if model_feature_names is None:
            model_feature_names = input_data.columns.tolist()
        model_feature_names = list(model_feature_names)

        explainer = shap.KernelExplainer(
            lambda X: model.predict_proba(pd.DataFrame(X, columns=model_feature_names)),
            bk_data_with_features.head(20)
        )
        shap_values = explainer.shap_values(input_data)

        sample_idx = 0
        # 兼容不同形状：list of arrays / 3D array
        if isinstance(shap_values, list):
            sample_shap = shap_values[1][sample_idx] if len(shap_values) == 2 else shap_values[0][sample_idx]
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        else:
            arr = np.asarray(shap_values)
            if arr.ndim == 3:
                sample_shap = arr[sample_idx][:, 1]
                base_value = explainer.expected_value[1]
            else:
                sample_shap = arr[sample_idx]
                base_value = explainer.expected_value

        sample_data = input_data.iloc[sample_idx]

        fig = shap.plots.force(
            base_value=base_value,
            shap_values=sample_shap,
            features=sample_data,
            feature_names=input_data.columns.tolist(),
            matplotlib=False,
            plot_cmap="coolwarm",
            text_rotation=15,
            figsize=(12, 6)
        )

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False)
        try:
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
        finally:
            tmp.close()
            if os.path.exists(tmp.name):
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass
    except Exception as e:
        st.error(f"SHAP plot generation failed: {str(e)}")
        st.stop()


def calculate_composite_indicator(indicator_name: str) -> None:
    if indicator_name not in COMPOSITE_INDICATORS:
        return
    info = COMPOSITE_INDICATORS[indicator_name]
    sub_values = []
    for sub_feat in info["sub_features"]:
        key = f"{indicator_name}_{sub_feat}_value"
        sub_values.append(st.session_state[key])
    composite_value = info["formula"](*sub_values)
    st.session_state[f"{indicator_name}_value"] = round(composite_value, 2)


def render_composite_popover(indicator: str, container):
    """通用复合指标 popover 组件"""
    with container:
        # 所有按钮统一为短标签 "📝 Edit XXX"，宽度一致 → 1 行不换行
        with st.popover(f"📝 Edit {indicator}", use_container_width=True):
            st.markdown(f"### {indicator} Sub-indicators")
            info = COMPOSITE_INDICATORS[indicator]
            for sub_feat, default_val in zip(info["sub_features"], info["default_values"]):
                key = f"{indicator}_{sub_feat}_value"
                st.number_input(
                    f"{sub_feat}",
                    step=0.01,
                    key=key,
                    on_change=calculate_composite_indicator,
                    args=(indicator,)
                )
            if st.button(f"Calculate {indicator}", key=f"calc_{indicator.lower()}"):
                calculate_composite_indicator(indicator)
                st.success(f"{indicator} = {st.session_state[f'{indicator}_value']}")

        # 用真正的 Streamlit widget（不设 key，只传 value）：
        # - 没 key -> 不走 session_state 缓存 -> 每次 rerun 都用最新的 value 刷新
        # - 是真正的 widget -> Streamlit 能测量高度 -> 不会与下方按钮重叠
        val = str(st.session_state[f"{indicator}_value"])
        st.text_input(
            label=f"{indicator}",
            value=val,
            disabled=True,
            help=f"Click 'Edit {indicator}' popover to recalculate",
        )
        return val


def main():
    ensure_reduced_scaler()
    model, scaler = load_model_and_scaler()

    with st.container():
        st.subheader("Enter Patient Data")

        # ---- Row 1: 5 basic features ----
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            age = st.number_input("Age", value=50.09, format="%.2f")
        with c2:
            sofa = st.number_input("SOFA", value=13.0, format="%.2f")
        with c3:
            Creatinine = st.number_input("Creatinine", value=1.3, format="%.2f")
        with c4:
            Bun = st.number_input("Bun", value=17.0, format="%.2f")
        with c5:
            Pt = st.number_input("Pt", value=12.5, format="%.2f")

        # ---- Row 2: 6 composite indicators (SII, NLR, PLR, NAR, MLR, APAR) ----
        d1, d2, d3, d4, d5, d6 = st.columns(6)
        SII = render_composite_popover("SII", d1)
        NLR = render_composite_popover("NLR", d2)
        PLR = render_composite_popover("PLR", d3)
        NAR = render_composite_popover("NAR", d4)
        MLR = render_composite_popover("MLR", d5)
        APAR = render_composite_popover("APAR", d6)

    inputs = {
        'admission_age': float(age),
        'sofa': float(sofa),
        'SII': float(SII),
        'NLR': float(NLR),
        'PLR': float(PLR),
        'NAR': float(NAR),
        'MLR': float(MLR),
        'APAR': float(APAR),
        'creatinine': float(Creatinine),
        'bun': float(Bun),
        'pt': float(Pt)
    }

    input_df = prepare_input_data(pd.DataFrame([inputs]), scaler)

    if st.button("Start Risk Assessment", type="primary"):
        try:
            # 只在 status 里做"计算过程"的进度反馈，结果本身放在外面
            with st.status("Analyzing...", expanded=True) as status:
                if input_df.isnull().any().any():
                    raise ValueError("Input data contains invalid values")

                st.write("Running model prediction...")
                risk = make_prediction(model, input_df)

                st.write("Computing SHAP explanations...")
                html_content = generate_shap_plot(model, input_df)

                status.update(label="Analysis complete", state="complete", expanded=False)

            # 结果放在 status 之外 —— 不会被任何 Streamlit 版本的折叠行为影响
            st.subheader("Risk Assessment Result")
            st.metric("Probability of Mortality", f"{risk * 100:.1f}%")

            st.subheader("Key Influencing Factors")
            st_html(html_content, height=600, scrolling=False)

        except Exception as e:
            print(str(e))
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
