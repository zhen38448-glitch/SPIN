import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats
import chardet
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams  
from sklearn.utils import resample
from scipy.interpolate import interp1d, interp2d, griddata, Rbf
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from scipy.optimize import curve_fit
import base64
from pathlib import Path


rcParams['font.sans-serif'] = ['SimHei']   
rcParams['axes.unicode_minus'] = False
# 设置页面
st.set_page_config(page_title="薄膜生长参数智能预测系统", layout="wide")

# 应用标题
st.title("🌱 薄膜生长参数智能预测系统")
st.markdown("使用机器学习分析薄膜生长参数与薄膜特征之间的关系，并预测最优工艺参数")

# 初始化session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'interpolated_data' not in st.session_state:
    st.session_state.interpolated_data = None
if 'interp_accum' not in st.session_state:
    st.session_state.interp_accum = pd.DataFrame()
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target' not in st.session_state:
    st.session_state.target = None
if 'feature_types' not in st.session_state:
    st.session_state.feature_types = {}
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'substrate_option' not in st.session_state:
    st.session_state.substrate_option = "所有基片"
if 'selected_substrate' not in st.session_state:
    st.session_state.selected_substrate = None
if 'augmentation_method' not in st.session_state:
    st.session_state.augmentation_method = "无"
if 'augmentation_factor' not in st.session_state:
    st.session_state.augmentation_factor = 1.0

# 读取本地图片并转换为 base64
def img_to_base64(img_path):
    return base64.b64encode(Path(img_path).read_bytes()).decode()

# 检测文件编码的函数
def detect_encoding(file_content):
    result = chardet.detect(file_content)
    return result['encoding']

# 数据增强函数
def augment_data(X, y, method="无", factor=1.0, feature_types=None):
    """
    数据增强函数
    method: 增强方法，可选 "无", "高斯噪声", "自助法", "插值法", "SMOTE式"
    factor: 增强因子，1.0表示不增强，2.0表示数据量翻倍
    """
    if method == "无" or factor <= 1.0:
        return X, y
    
    n_samples = len(X)
    n_new = int(n_samples * (factor - 1.0))
    
    if n_new <= 0:
        return X, y
    
    X_new = X.copy()
    y_new = y.copy()
    
    if method == "高斯噪声":
        # 添加高斯噪声
        numeric_cols = [col for col in X.columns if feature_types.get(col) == "数值型"]
        categorical_cols = [col for col in X.columns if feature_types.get(col) == "分类型"]
        
        for _ in range(n_new):
            # 随机选择一个样本
            idx = np.random.randint(0, n_samples)
            sample = X.iloc[idx].copy()
            
            # 对数值型特征添加噪声
            for col in numeric_cols:
                std = X[col].std() * 0.05  # 噪声标准差为原特征标准差的5%
                sample[col] += np.random.normal(0, std)
            
            X_new = pd.concat([X_new, sample.to_frame().T], ignore_index=True)
            y_new = pd.concat([y_new, pd.Series([y.iloc[idx]])], ignore_index=True)
    
    elif method == "自助法":
        # 自助法重采样
        for _ in range(n_new):
            idx = np.random.randint(0, n_samples)
            X_new = pd.concat([X_new, X.iloc[[idx]]], ignore_index=True)
            y_new = pd.concat([y_new, pd.Series([y.iloc[idx]])], ignore_index=True)
    
    elif method == "插值法":
        # 在数值型特征之间进行插值
        numeric_cols = [col for col in X.columns if feature_types.get(col) == "数值型"]
        categorical_cols = [col for col in X.columns if feature_types.get(col) == "分类型"]
        
        for _ in range(n_new):
            # 随机选择两个不同的样本
            idx1, idx2 = np.random.choice(n_samples, 2, replace=False)
            alpha = np.random.random()
            
            # 创建新样本
            new_sample = X.iloc[idx1].copy()
            
            # 对数值型特征进行插值
            for col in numeric_cols:
                val1 = X.iloc[idx1][col]
                val2 = X.iloc[idx2][col]
                new_sample[col] = alpha * val1 + (1 - alpha) * val2
            
            # 对分类特征，随机选择其中一个样本的值
            for col in categorical_cols:
                if np.random.random() > 0.5:
                    new_sample[col] = X.iloc[idx1][col]
                else:
                    new_sample[col] = X.iloc[idx2][col]
            
            # 对目标变量进行插值
            new_target = alpha * y.iloc[idx1] + (1 - alpha) * y.iloc[idx2]
            
            X_new = pd.concat([X_new, new_sample.to_frame().T], ignore_index=True)
            y_new = pd.concat([y_new, pd.Series([new_target])], ignore_index=True)
    
    elif method == "SMOTE式":
        # 类似SMOTE的方法，用于回归问题
        numeric_cols = [col for col in X.columns if feature_types.get(col) == "数值型"]
        categorical_cols = [col for col in X.columns if feature_types.get(col) == "分类型"]
        
        # 使用K近邻
        n_neighbors = min(5, n_samples - 1)
        if n_neighbors > 0:
            knn = NearestNeighbors(n_neighbors=n_neighbors)
            knn.fit(X[numeric_cols])
            
            for _ in range(n_new):
                # 随机选择一个样本
                idx = np.random.randint(0, n_samples)
                
                # 找到最近邻
                distances, indices = knn.kneighbors([X.iloc[idx][numeric_cols]], n_neighbors+1)
                
                # 随机选择一个近邻（排除自身）
                neighbor_idx = np.random.choice(indices[0][1:])
                
                # 创建新样本
                new_sample = X.iloc[idx].copy()
                alpha = np.random.random()
                
                # 对数值型特征进行插值
                for col in numeric_cols:
                    val1 = X.iloc[idx][col]
                    val2 = X.iloc[neighbor_idx][col]
                    new_sample[col] = alpha * val1 + (1 - alpha) * val2
                
                # 对目标变量进行插值
                new_target = alpha * y.iloc[idx] + (1 - alpha) * y.iloc[neighbor_idx]
                
                X_new = pd.concat([X_new, new_sample.to_frame().T], ignore_index=True)
                y_new = pd.concat([y_new, pd.Series([new_target])], ignore_index=True)
    
    return X_new, y_new

# 插值函数
def interpolate_scalar(val1, val2, alpha, method='linear'):
    # 对单值的插值器（数值）
    if method == 'linear':
        return (1 - alpha) * val1 + alpha * val2
    elif method == 'quadratic':
        t = alpha
        return (1 - t)**2 * val1 + 2 * (1 - t) * t * ((val1 + val2) / 2) + t**2 * val2
    elif method == 'exponential':
        if val1 > 0 and val2 > 0:
            return val1 * (val2 / val1) ** alpha
        else:
            return (1 - alpha) * val1 + alpha * val2
    elif method == 'logarithmic':
        if val1 > 0 and val2 > 0:
            return val1 * np.exp(alpha * np.log(val2 / val1))
        else:
            return (1 - alpha) * val1 + alpha * val2
    else:
        return (1 - alpha) * val1 + alpha * val2
    
def interpolate_between_points(point1, point2, feature_names, target_name, num_points=5, method='linear', feature_types=None):
    """
    在两点之间进行插值
    """
    # 允许外部传入 feature_types；若未传入则回退到 session_state
    if feature_types is None:
        feature_types = st.session_state.feature_types if 'feature_types' in st.session_state else {}
    
    # 提取数值型特征
    numeric_features = [f for f in feature_names if f != target_name and st.session_state.feature_types.get(f) == "数值型"]
    categorical_features = [f for f in feature_names if f != target_name and st.session_state.feature_types.get(f) == "分类型"]
    
    # 创建插值结果
    interpolated_points = []
    
    # 对每个插值点
    for i in range(num_points):
        alpha = i / (num_points - 1) if num_points > 1 else 0.5
        
        # 创建新点
        new_point = {}
        
        # 对数值型特征进行插值
        for feature in numeric_features:
            val1 = point1[feature]
            val2 = point2[feature]
            
            if method == 'linear':
                new_point[feature] = (1 - alpha) * val1 + alpha * val2
            elif method == 'quadratic':
                # 二次插值
                t = alpha
                new_point[feature] = (1 - t)**2 * val1 + 2 * (1 - t) * t * ((val1 + val2) / 2) + t**2 * val2
            elif method == 'exponential':
                # 指数插值
                if val1 > 0 and val2 > 0:
                    new_point[feature] = val1 * (val2 / val1) ** alpha
                else:
                    new_point[feature] = (1 - alpha) * val1 + alpha * val2
            elif method == 'logarithmic':
                # 对数插值
                if val1 > 0 and val2 > 0:
                    new_point[feature] = val1 * np.exp(alpha * np.log(val2 / val1))
                else:
                    new_point[feature] = (1 - alpha) * val1 + alpha * val2
        
        # 对分类特征，选择第一个点的值
        for feature in categorical_features:
            new_point[feature] = point1[feature]
        
        # 对目标变量进行插值
        val1 = point1[target_name]
        val2 = point2[target_name]
        
        if method == 'linear':
            new_point[target_name] = (1 - alpha) * val1 + alpha * val2
        elif method == 'quadratic':
            t = alpha
            new_point[target_name] = (1 - t)**2 * val1 + 2 * (1 - t) * t * ((val1 + val2) / 2) + t**2 * val2
        elif method == 'exponential':
            if val1 > 0 and val2 > 0:
                new_point[target_name] = val1 * (val2 / val1) ** alpha
            else:
                new_point[target_name] = (1 - alpha) * val1 + alpha * val2
        elif method == 'logarithmic':
            if val1 > 0 and val2 > 0:
                new_point[target_name] = val1 * np.exp(alpha * np.log(val2 / val1))
            else:
                new_point[target_name] = (1 - alpha) * val1 + alpha * val2
        
        interpolated_points.append(new_point)
    
    return interpolated_points

def visualize_xy(x, y, x_label, y_label, title, orig=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    if orig is not None:
        ax.scatter(orig['x'], orig['y'], s=120, c='red', label='原始节点')
    ax.plot(x, y, 'o--', alpha=0.7, label='插值序列')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
logo_path = "logo.png"  # 替换为您的图片路径
logo_base64 = img_to_base64(logo_path)
st.sidebar.markdown(
    f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="400"></div>', 
    unsafe_allow_html=True
)
def _load_model_from_joblib(uploaded_file):
    """
    读取 joblib/pkl。兼容两种保存方式：
    1) dict 包含 {"model":..., "target":..., "features":..., "feature_types":...}
    2) 直接是拟合好的 sklearn Pipeline/Estimator
    返回: model, target, features, feature_types
    """
    obj = joblib.load(uploaded_file)
    model, target, features, feature_types = obj, None, None, None
    if isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        target = obj.get("target", None)
        features = obj.get("features", None)
        feature_types = obj.get("feature_types", None)
    return model, target, features, feature_types

def _infer_features_if_missing(model, fallback_df, known_targets=None):
    """
    尝试在缺失 features 时推断。优先：
    1) sklearn >=1.0 的 feature_names_in_
    2) 从 fallback_df 去掉已知 targets
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if fallback_df is not None:
        known_targets = set(known_targets or [])
        return [c for c in fallback_df.columns if c not in known_targets]
    return []

def _infer_feature_types_if_missing(features, df, given=None):
    """
    在未提供 feature_types 时，从 df.dtype 推断：数值型/分类型
    """
    if given:
        return given
    ft = {}
    if df is None:
        return ft
    for f in features:
        if f in df.columns:
            ft[f] = "数值型" if str(df[f].dtype) in ["int64", "float64"] else "分类型"
    return ft

def _sample_candidates(n, features, feature_types, param_limits):
    """
    在给定限制 param_limits 下随机生成 n 组候选参数。
    param_limits:
      - 数值型: (low, high)
      - 分类型: [opt1, opt2, ...]
    """
    rows = []
    for _ in range(n):
        row = {}
        for f in features:
            if feature_types.get(f) == "数值型":
                low, high = param_limits[f]
                row[f] = np.random.uniform(low, high)
            else:
                choices = param_limits[f]
                # 防御：若空列表，先跳过，稍后用全部 unique 值兜底
                row[f] = np.random.choice(choices) if len(choices) > 0 else None
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_with_models(candidates_df, models_by_target, global_features=None):
    """
    使用“多个单输出模型（每个 target 一个）”进行预测。
    models_by_target: { target: {"model":..., "features":..., "feature_types":...} }
    global_features: 若单模型条目标未携带 features，则用此全局 features。
    返回：preds_df（列为各 target）
    """
    preds = {}
    for t, pack in models_by_target.items():
        m = pack["model"]
        fts = pack.get("features")
        if fts is None or len(fts) == 0:
            # 从模型或全局推断
            fts = _infer_features_if_missing(m, st.session_state.get("data"), known_targets=[t])
            if (not fts) and global_features:
                fts = list(global_features)
        # 防御：若 candidates 不含该模型全部特征，做交集
        use_cols = [c for c in fts if c in candidates_df.columns]
        X = candidates_df[use_cols]
        yhat = m.predict(X)
        # 统一成 1D
        yhat = np.ravel(yhat)
        preds[t] = yhat
    return pd.DataFrame(preds)


def _compute_weighted_error(preds_df, target_vals, target_weights=None):
    """
    计算加权 L1 误差。target_weights 可为 {target: weight}
    """
    if target_weights is None:
        target_weights = {t: 1.0 for t in target_vals.keys()}
    err = 0.0
    for t, val in target_vals.items():
        w = float(target_weights.get(t, 1.0))
        err += w * np.abs(preds_df[t] - val)
    return err

# 侧边栏导航
page = st.sidebar.selectbox("导航", ["实验数据插值增强","数据上传与设置", "模型训练", "参数预测","反向预测", "特征分析", "参数分布分析","模型稳定性验证"])
# -------------------- 页面：实验数据插值增强 --------------------
if page == "实验数据插值增强":
    st.header("🔬 小样本插值构造器（基于物理直觉的数据增广）")
    st.markdown(
        "适用于 5~10 组小样本：任意选择两点；"
        "当**单变量**变化时选择插值函数；当**多变量**变化时，设定变化顺序与过渡节点（可给定节点目标经验值），"
        "为每个阶段分别选择插值方法与点数；结果可**累计到插值缓存**并**导出/追加至CSV**。"
    )

    # 读取小样本数据
    uploaded_file = st.file_uploader("上传小样本 CSV", type=['csv'])
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read()
            encoding = detect_encoding(file_content)
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            if df.empty or df.columns.str.contains('Unnamed').any():
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, header=0)

            st.session_state.data = df
            st.subheader("数据预览")
            st.dataframe(df)
            st.caption(f"编码: {encoding}，形状: {df.shape[0]} 行 × {df.shape[1]} 列")

            # 选择目标与特征，并标注类型
            target = st.selectbox("选择预测目标(观测量)", df.columns.tolist(), key="interp_target")
            feature_options = [c for c in df.columns if c != target]
            selected_features = st.multiselect("选择特征参数", feature_options, default=feature_options, key="interp_features")

            st.subheader("标注特征类型")
            feature_types = {}
            cols2 = st.columns(2)
            for i, f in enumerate(selected_features):
                with cols2[i % 2]:
                    default_type = "数值型" if str(df[f].dtype) in ['int64', 'float64'] else "分类型"
                    feature_types[f] = st.selectbox(f"{f} 类型", ["数值型", "分类型"],
                                                    index=0 if default_type == "数值型" else 1,
                                                    key=f"interp_type_{f}")
            st.session_state.feature_types = feature_types

            if len(df) < 2:
                st.error("至少需要两个数据点。")
                st.stop()

            st.subheader("选择两点")
            idx1 = st.selectbox("第一个点（索引）", list(range(len(df))), format_func=lambda i: f"#{i+1}")
            idx2 = st.selectbox("第二个点（索引）", list(range(len(df))), index=min(1, len(df)-1), format_func=lambda i: f"#{i+1}")
            if idx1 == idx2:
                st.error("请选择两个不同的数据点。")
                st.stop()

            p1 = df.iloc[idx1].to_dict()
            p2 = df.iloc[idx2].to_dict()
            # ===== 两点详情（横向表格）【替换原先使用 st.json 的 expander 块】=====
            with st.expander("两点详情（横向表格）", expanded=True):
                cols_show = selected_features + [target]

                # 直接从 df 按行切片，保留"横向"结构，避免 dict.name 报错
                row1 = df.loc[[idx1], cols_show].copy()
                row2 = df.loc[[idx2], cols_show].copy()
                row1.index = [f"点 #{idx1+1}"]
                row2.index = [f"点 #{idx2+1}"]

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**点 1**")
                    st.dataframe(row1, use_container_width=True)
                with c2:
                    st.markdown("**点 2**")
                    st.dataframe(row2, use_container_width=True)

                # 可选：合并对比 + 高亮出"不同"的列（黄色）
                compare_df = pd.concat([row1, row2], axis=0)

                def _style_diff(df_):
                    styles = pd.DataFrame('', index=df_.index, columns=df_.columns)
                    if len(df_) == 2:
                        a, b = df_.iloc[0], df_.iloc[1]
                        for c in df_.columns:
                            v1, v2 = a[c], b[c]
                            different = False
                            if pd.api.types.is_numeric_dtype(df_[c]):
                                if pd.isna(v1) or pd.isna(v2):
                                    different = pd.isna(v1) != pd.isna(v2)
                                else:
                                    try:
                                        different = not np.isclose(float(v1), float(v2), rtol=1e-6, atol=1e-12)
                                    except Exception:
                                        different = (v1 != v2)
                            else:
                                different = (v1 != v2)
                            if different:
                                styles.loc[:, c] = 'background-color: #ffeb3b33'  # 浅黄
                    return styles

                st.caption("两点合并对比（黄色=两点取值不同）")
                try:
                    st.dataframe(compare_df.style.apply(_style_diff, axis=None), use_container_width=True)
                except Exception:
                    # 若本地 Streamlit 版本不支持 Styler，退化为普通表格
                    st.dataframe(compare_df, use_container_width=True)

            
            # 检测变化特征
            changed = [f for f in selected_features if p1[f] != p2[f]]
            st.info(f"变化的特征：{', '.join(changed) if changed else '无'}")

            # ---------- 单变量 ----------
            if len(changed) == 1:
                st.success("✅ 单变量变化，可直接插值")
                m1, m2, m3 = st.columns([1,1,1])
                with m1:
                    num_points = st.slider("插值点数（含两端）", 2, 50, 7)
                with m2:
                    method_name = st.selectbox("插值方法", ["线性", "二次", "指数", "对数"])
                method_map = {"线性":"linear", "二次":"quadratic", "指数":"exponential", "对数":"logarithmic"}
                method = method_map[method_name]

                if st.button("生成插值数据（单变量）", use_container_width=True):
                    interps = interpolate_between_points(
                        p1, p2, selected_features + [target], target,
                        num_points=num_points, method=method, feature_types=feature_types
                    )
                    interp_df = pd.DataFrame(interps)

                    st.subheader("插值结果（单变量）")
                    st.dataframe(interp_df)

                    # 可视化（x=唯一变化特征）
                    xf = changed[0]
                    x = [d[xf] for d in interps]
                    y = [d[target] for d in interps]
                    visualize_xy(x, y, xf, target, f"{xf} → {target} ({method_name})",
                                 orig={'x':[p1[xf], p2[xf]], 'y':[p1[target], p2[target]]})

                    # 累计到插值缓存
                    st.session_state.interpolated_data = interp_df
                    st.session_state.interp_accum = pd.concat([st.session_state.interp_accum, interp_df], ignore_index=True)
                    st.success(f"已追加到插值缓存，当前缓存共 {len(st.session_state.interp_accum)} 行。")

            # ---------- 多变量 ----------
            elif len(changed) >= 2:
                st.warning("⚠️ 多变量变化：请设置变化顺序、过渡节点与每阶段方法。")

                # 变化顺序（排列）
                order = st.multiselect("选择并排序变量变化顺序（从先到后）", changed, default=changed)
                if len(order) != len(changed):
                    st.error("请把所有发生变化的特征都放入顺序列表，并保持无重复。")
                    st.stop()

                # 过渡节点数 = n-1
                n = len(changed)
                num_nodes = n - 1
                st.caption(f"将从 {tuple(p1[f] for f in order)} 变化到 {tuple(p2[f] for f in order)}，需要 {num_nodes} 个过渡节点。")
                st.info("节点含义：按顺序依次把特征从点1的值切换成点2的值，例如 A→B→C：节点1改变 A，节点2 再改变 B，最终第三段改变 C。")

                # 生成节点（仅特征，不含 target）
                nodes = []
                cur = p1.copy()
                for i, feat in enumerate(order):
                    # 第 i 个节点：feat 改成 p2[feat]，其它未变动的仍取当前值
                    cur = cur.copy()
                    cur[feat] = p2[feat]
                    nodes.append({f: cur[f] for f in selected_features})

                # 让用户可设置每个**中间节点**的经验目标值（不含起点、终点）
                st.subheader("设置节点目标经验值（可选）")
                st.caption("如果不填，则默认按整段线性过渡估计节点目标值。")
                node_targets = {}
                for i in range(num_nodes):
                    with st.expander(f"过渡节点 #{i+1}（完成 {order[i]} 的切换）", expanded=False):
                        default_alpha = (i+1) / (n)  # 在 0..1 均匀估计
                        default_t = interpolate_scalar(p1[target], p2[target], default_alpha, method='linear')
                        val = st.text_input(
                            f"节点 #{i+1} 的 {target} 经验值",
                            value="",
                            help=f"留空表示使用默认估计值（当前建议 {default_t:.4f}）",
                            key=f"node_t_{i}"
                        )
                        node_targets[i] = float(val) if val.strip() != "" else float(default_t)
                st.session_state.node_targets = node_targets

                # ===== 过渡节点状态表（实时）【插在"设置节点目标经验值"之后，"每阶段配置"之前】=====
                st.subheader("过渡节点状态表（实时）")
                node_rows = []
                node_labels = []

                # 起点
                node_rows.append({**{f: p1[f] for f in selected_features}, target: p1[target]})
                node_labels.append("起点")

                # 中间过渡节点：使用上方输入框设置的经验目标值（若留空则采用默认线性估计）
                for i in range(num_nodes):
                    r = nodes[i].copy()
                    r[target] = st.session_state.node_targets[i]
                    node_rows.append(r)
                    node_labels.append(f"过渡{i+1}")

                # 终点
                node_rows.append({**{f: p2[f] for f in selected_features}, target: p2[target]})
                node_labels.append("终点")

                node_table = pd.DataFrame(node_rows, index=node_labels)[selected_features + [target]]

                def _style_transition(df_):
                    # 高亮每行相对于上一行发生变化的单元格（浅绿）
                    styles = pd.DataFrame('', index=df_.index, columns=df_.columns)
                    for i in range(1, len(df_)):
                        prev = df_.iloc[i-1]
                        cur = df_.iloc[i]
                        for c in df_.columns:
                            changed_cell = False
                            if pd.api.types.is_numeric_dtype(df_[c]):
                                v1, v2 = prev[c], cur[c]
                                if pd.isna(v1) or pd.isna(v2):
                                    changed_cell = pd.isna(v1) != pd.isna(v2)
                                else:
                                    try:
                                        changed_cell = not np.isclose(float(v1), float(v2), rtol=1e-6, atol=1e-12)
                                    except Exception:
                                        changed_cell = (v1 != v2)
                            else:
                                changed_cell = (prev[c] != cur[c])

                            if changed_cell:
                                styles.iloc[i, df_.columns.get_loc(c)] = 'background-color: #87CEFA'  # 浅绿
                    return styles

                try:
                    st.dataframe(node_table.style.apply(_style_transition, axis=None), use_container_width=True)
                except Exception:
                    st.dataframe(node_table, use_container_width=True)

                st.caption("浅绿表示该节点相对于上一节点在该字段发生了变化；上方输入的观测量经验值会即时反映到表格中。")


                st.subheader("每阶段配置（方法 & 插值点数）")
                stage_cfg = {}
                stage_cols = st.columns(3)
                for s in range(n):
                    with stage_cols[s % 3]:
                        st.markdown(f"**阶段 {s+1}：改变 {order[s]}**")
                        mname = st.selectbox(f"方法（阶段 {s+1}）", ["线性", "二次", "指数", "对数"], key=f"stage_m_{s}")
                        pts = st.number_input(f"点数（阶段 {s+1}，含端点）", 2, 50, 5, key=f"stage_n_{s}")
                        stage_cfg[s] = {"method": {"线性":"linear","二次":"quadratic","指数":"exponential","对数":"logarithmic"}[mname],
                                        "num_points": int(pts)}
                st.session_state.stage_cfg = stage_cfg

                if st.button("生成插值数据（多变量阶段式）", use_container_width=True):
                    # 构建节点列表（含起点、过渡节点、终点）并注入 target
                    node_list = [ {**{f:p1[f] for f in selected_features}, target: p1[target]} ]
                    for i in range(num_nodes):
                        node = nodes[i].copy()
                        node[target] = st.session_state.node_targets[i]
                        node_list.append(node)
                    node_list.append( {**{f:p2[f] for f in selected_features}, target: p2[target]} )

                    # 按 node_list 相邻两点进行阶段式插值
                    all_points = []
                    for s in range(len(node_list)-1):
                        a = node_list[s]
                        b = node_list[s+1]
                        cfg = st.session_state.stage_cfg[s]
                        pts = cfg["num_points"]
                        meth = cfg["method"]
                        seg = interpolate_between_points(
                            a, b, selected_features + [target], target,
                            num_points=pts, method=meth, feature_types=feature_types
                        )
                        # 连接时避免重复中间节点（去掉段内第一个点，保留第一个段的第一个点）
                        if s > 0 and len(seg) > 0:
                            seg = seg[1:]
                        all_points.extend(seg)

                    final_df = pd.DataFrame(all_points)
                    st.subheader("阶段式插值结果")
                    st.dataframe(final_df)

                    # 多图可视化：每个数值特征对 target 的阶段结果（按节点顺序）
                    st.subheader("可视化：各数值特征与目标的插值轨迹")
                    numeric_features = [f for f in selected_features if feature_types.get(f) == "数值型"]
                    for f in numeric_features:
                        x = final_df[f].tolist()
                        y = final_df[target].tolist()
                        visualize_xy(x, y, f, target, f"{f} → {target}（阶段式）",
                                     orig={'x':[p1[f], p2[f]], 'y':[p1[target], p2[target]]})

                    # 保存到状态 & 累计缓存
                    st.session_state.interpolated_data = final_df
                    st.session_state.interp_accum = pd.concat([st.session_state.interp_accum, final_df], ignore_index=True)
                    st.success(f"已追加到插值缓存，当前缓存共 {len(st.session_state.interp_accum)} 行。")

            else:
                st.info("两点在所选特征上并无差异，无需插值。")

            # ---------- 插值缓存区 & 导出/追加 ----------
            st.subheader("📦 插值缓存")
            if not st.session_state.interp_accum.empty:
                st.dataframe(st.session_state.interp_accum.tail(50))
                cexp1, cexp2, cexp3 = st.columns([1,1,1])

                with cexp1:
                    # 新文件导出
                    fn = st.text_input("导出为（新文件名）", "interpolated_accum.csv")
                    if st.button("导出当前缓存为新CSV", use_container_width=True):
                        csv = st.session_state.interp_accum.to_csv(index=False).encode('utf-8')
                        st.download_button("下载 CSV", csv, file_name=fn, mime="text/csv", use_container_width=True)

                with cexp2:
                    # 追加到已有 CSV（上传已有文件，返回合并后的下载）
                    append_file = st.file_uploader("选择一个已有CSV以追加", type=['csv'], key="append_csv")
                    if append_file is not None:
                        base = pd.read_csv(append_file)
                        merged = pd.concat([base, st.session_state.interp_accum], ignore_index=True)
                        st.write("预览合并结果（尾部）")
                        st.dataframe(merged.tail(50))
                        csvm = merged.to_csv(index=False).encode('utf-8')
                        st.download_button("下载合并后的 CSV", csvm, file_name="merged_appended.csv", mime="text/csv", use_container_width=True)

                with cexp3:
                    if st.button("清空插值缓存", use_container_width=True, type="secondary"):
                        st.session_state.interp_accum = pd.DataFrame()
                        st.success("缓存已清空。")

        except Exception as e:
            st.error(f"读取文件时出错: {str(e)}")
            st.info("请尝试将文件另存为 UTF-8 编码后重新上传。")

    else:
        st.info("请先上传 5~10 组小样本 CSV 开始构造插值数据。")
    
# 页面1: 数据上传与设置
elif page == "数据上传与设置":
    st.header("📊 数据上传与特征设置")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传CSV数据文件", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # 读取文件内容并检测编码
            file_content = uploaded_file.read()
            encoding = detect_encoding(file_content)
            
            # 重置文件指针
            uploaded_file.seek(0)
            
            # 使用检测到的编码读取CSV
            df = pd.read_csv(uploaded_file, encoding=encoding)
            
            # 尝试处理可能的中文列名问题
            if df.empty or df.columns.str.contains('Unnamed').any():
                # 如果列名有问题，尝试重新读取
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding, header=0)
                
            st.session_state.data = df
            
            # 显示数据基本信息
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            st.write(f"数据集形状: {df.shape[0]} 行, {df.shape[1]} 列")
            st.write(f"检测到的文件编码: {encoding}")
            
            # 选择目标变量
            target_options = df.columns.tolist()
            st.session_state.target = st.selectbox("选择预测目标(观测量)", target_options)
            
            # 选择特征
            feature_options = [col for col in df.columns if col != st.session_state.target]
            selected_features = st.multiselect("选择特征参数", feature_options, default=feature_options)
            st.session_state.features = selected_features
            
            # 设置特征类型
            st.subheader("设置特征类型")
            st.write("请为每个特征指定类型（数值型或分类型）")
            
            feature_types = {}
            cols = st.columns(2)
            for i, feature in enumerate(selected_features):
                with cols[i % 2]:
                    # 尝试自动判断类型
                    if df[feature].dtype in ['int64', 'float64']:
                        default_type = "数值型"
                    else:
                        default_type = "分类型"
                    
                    feature_type = st.selectbox(
                        f"{feature} 类型", 
                        ["数值型", "分类型"], 
                        index=0 if default_type == "数值型" else 1,
                        key=f"type_{feature}"
                    )
                    feature_types[feature] = feature_type
            
            st.session_state.feature_types = feature_types
            
            # 数据增强设置
            st.subheader("数据增强设置")
            st.write("小数据集建议使用数据增强提高模型稳定性")
            
            augmentation_method = st.selectbox(
                "数据增强方法",
                ["无", "高斯噪声", "自助法", "插值法", "SMOTE式"],
                help="小数据量推荐：高斯噪声、插值法、SMOTE式"
            )
            
            augmentation_factor = st.slider(
                "数据增强倍数", 
                1.0, 100.0, 1.5, 0.1,
                help="1.0表示不增强，2.0表示数据量翻倍"
            )
            
            st.session_state.augmentation_method = augmentation_method
            st.session_state.augmentation_factor = augmentation_factor
            
            # 基片类型选择
            st.subheader("基片类型设置")
            
            # 检查是否有基片类型的特征
            substrate_features = [f for f in selected_features if "substrate" in f.lower() or "基片" in f.lower()]
            
            if substrate_features:
                substrate_feature = substrate_features[0]  # 使用第一个基片相关特征
                substrate_options = df[substrate_feature].unique().tolist()
                
                st.session_state.substrate_option = st.radio(
                    "选择基片数据处理方式",
                    ["所有基片", "特定基片"],
                    help="选择是否针对特定基片类型训练模型"
                )
                
                if st.session_state.substrate_option == "特定基片":
                    st.session_state.selected_substrate = st.selectbox(
                        "选择基片类型", 
                        substrate_options
                    )
                
                st.session_state.substrate_feature = substrate_feature
            else:
                st.info("未检测到基片类型特征。如果您有基片类型数据，请确保特征名称中包含'substrate'或'基片'")
                st.session_state.substrate_option = "所有基片"
                st.session_state.selected_substrate = None
            
            st.success("数据设置完成！请转到「模型训练」页面训练模型")
            
        except Exception as e:
            st.error(f"读取文件时出错: {str(e)}")
            st.info("请尝试将文件另存为UTF-8编码格式后重新上传")

# 页面2: 模型训练
elif page == "模型训练":
    st.header("🤖 模型训练")
    
    if st.session_state.data is None:
        st.warning("请先在「数据上传与设置」页面上传数据并设置参数")
    else:
        df = st.session_state.data
        target = st.session_state.target
        features = st.session_state.features
        feature_types = st.session_state.feature_types
        
        # 根据基片选择过滤数据
        if st.session_state.substrate_option == "特定基片" and st.session_state.selected_substrate:
            substrate_feature = st.session_state.substrate_feature
            df = df[df[substrate_feature] == st.session_state.selected_substrate]
            st.write(f"使用基片类型: **{st.session_state.selected_substrate}**")
            st.write(f"筛选后数据量: {len(df)} 条")
        
        st.write(f"目标变量: **{target}**")
        st.write(f"特征变量: {', '.join(features)}")
        
        if len(df) < 10:
            st.error("数据量过少，无法进行有效的模型训练。请选择其他基片类型或使用所有基片数据。")
        else:
            # 划分数据
            test_size = st.slider("测试集比例", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("随机种子", 0, 100, 42)
            
            X = df[features]
            y = df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 数据增强
            st.subheader("数据增强")
            augmentation_method = st.session_state.augmentation_method
            augmentation_factor = st.session_state.augmentation_factor
            
            if augmentation_method != "无":
                st.write(f"使用数据增强方法: **{augmentation_method}**")
                st.write(f"增强倍数: **{augmentation_factor}**")
                
                original_size = len(X_train)
                X_train, y_train = augment_data(
                    X_train, y_train, 
                    method=augmentation_method,
                    factor=augmentation_factor,
                    feature_types=feature_types
                )
                new_size = len(X_train)
                
                st.write(f"训练数据从 {original_size} 条增强到 {new_size} 条")
          
            # 构建预处理管道
            numeric_features = [f for f in features if feature_types[f] == "数值型"]
            categorical_features = [f for f in features if feature_types[f] == "分类型"]
            
            # 如果基片特征已经在分类特征中，确保它不会被重复处理
            if hasattr(st.session_state, 'substrate_feature') and st.session_state.substrate_feature in categorical_features:
                categorical_features = [f for f in categorical_features if f != st.session_state.substrate_feature]
            
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])
            
            # 模型选择
            st.subheader("模型选择与参数设置")
            model_type = st.selectbox(
                "选择机器学习算法",
                ["随机森林回归", "线性回归", "Ridge回归", "Lasso回归", "支持向量回归(SVR)"],
                help="小数据量推荐：线性回归、Ridge回归、Lasso回归\n大数据量推荐：随机森林、SVR"
            )
            
            # 根据选择的模型类型设置相应的超参数
            model_params = {}
            if model_type == "随机森林回归":
                model_params['n_estimators'] = st.slider("树的数量", 10, 500, 100, 10)
                model_params['max_depth'] = st.slider("最大深度", 1, 50, None, 1)
            elif model_type in ["Ridge回归", "Lasso回归"]:
                model_params['alpha'] = st.slider("正则化强度", 0.01, 10.0, 1.0, 0.01)
                model_params['max_iter'] = st.slider("最大迭代次数", 100, 10000, 1000, 100)
            elif model_type == "支持向量回归(SVR)":
                model_params['kernel'] = st.selectbox("核函数", ["linear", "poly", "rbf", "sigmoid"], index=2)
                model_params['C'] = st.slider("惩罚参数C", 0.1, 100.0, 1.0, 0.1)
                model_params['gamma'] = st.selectbox("核函数系数gamma", ["scale", "auto"], index=0) 
            # 训练完成后保存增强数据和预测结果到session_state
            st.session_state.X_train_augmented = X_train
            st.session_state.y_train_augmented = y_train
            st.session_state.X_test = X_test
            # 训练模型按钮
            if st.button("训练模型"):
                with st.spinner("模型训练中..."):
                    # 根据选择的模型类型创建相应的回归器
                    if model_type == "随机森林回归":
                        regressor = RandomForestRegressor(
                            n_estimators=model_params['n_estimators'],
                            max_depth=model_params['max_depth'],
                            random_state=random_state
                        )
                    elif model_type == "线性回归":
                        from sklearn.linear_model import LinearRegression
                        regressor = LinearRegression()
                    elif model_type == "Ridge回归":
                        from sklearn.linear_model import Ridge
                        regressor = Ridge(
                            alpha=model_params['alpha'],
                            max_iter=model_params['max_iter'],
                            random_state=random_state
                        )
                    elif model_type == "Lasso回归":
                        from sklearn.linear_model import Lasso
                        regressor = Lasso(
                            alpha=model_params['alpha'],
                            max_iter=model_params['max_iter'],
                            random_state=random_state
                        )
                    elif model_type == "支持向量回归(SVR)":
                        # 确保所有必要的参数都已设置
                        if 'kernel' not in model_params:
                            model_params['kernel'] = 'rbf'  # 设置默认值
                        if 'C' not in model_params:
                            model_params['C'] = 1.0  # 设置默认值
                        if 'gamma' not in model_params:
                            model_params['gamma'] = 'scale'  # 设置默认值
                        
                        regressor = SVR(
                            kernel=model_params['kernel'],
                            C=model_params['C'],
                            gamma=model_params['gamma']
                        )
                    
                    # 建立模型管道
                    model = Pipeline([
                        ('preprocessor', preprocessor),
                        ('regressor', regressor)
                    ])
                    
                    # 训练模型
                    model.fit(X_train, y_train)
                    
                    # 评估模型
                    # 评估模型
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # 保存模型到session state
                    st.session_state.model = model
                    st.session_state.trained = True
                    st.session_state.model_type = model_type  # 保存模型类型用于后续分析
                    st.session_state.y_pred = y_pred
                    # 显示结果
                    st.success(f"模型训练完成！使用算法: {model_type}")
                    st.metric("R² 分数", f"{r2:.4f}")
                    st.metric("RMSE", f"{rmse:.4f}")
                    
                    # 真实值 vs 预测值图
                    fig, ax = plt.subplots()
                    ax.scatter(y_test, y_pred, alpha=0.5)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
                    ax.set_xlabel('真实值')
                    ax.set_ylabel('预测值')
                    ax.set_title('真实值 vs 预测值')
                    st.pyplot(fig)
                    
                    # 改进的模型保存功能
                    if st.session_state.trained:
                        st.subheader("💾 模型保存")
                        
                        # 创建两列布局
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # 提供默认文件名建议
                            default_filename = f"{st.session_state.model_type}_{st.session_state.target}_model.joblib"
                            model_filename = st.text_input("模型保存文件名", value=default_filename)
                        
                        with col2:
                            st.markdown("<br>", unsafe_allow_html=True)  # 垂直对齐
                            # 添加文件格式说明
                            st.caption("文件格式: .joblib")
                        
                        # 添加模型信息摘要
                        st.info(f"""
                        **模型信息摘要:**
                        - 模型类型: {st.session_state.model_type}
                        - 预测目标: {st.session_state.target}
                        - 特征数量: {len(st.session_state.features)}
                        - 训练数据量: {len(st.session_state.X_train_augmented) if hasattr(st.session_state, 'X_train_augmented') else '未知'}
                        """)
                        
                        # 将模型转换为字节流
                        try:
                            model_bytes = io.BytesIO()
                            joblib.dump({"model": st.session_state.model, "target": st.session_state.target}, model_bytes)
                            model_bytes.seek(0)
                            
                            # 添加文件大小信息
                            file_size = len(model_bytes.getvalue())
                            size_mb = file_size / (1024 * 1024)
                            st.caption(f"预计文件大小: {size_mb:.2f} MB")
                            
                            # 提供下载按钮
                            st.download_button(
                                label="📥 下载训练好的模型",
                                data=model_bytes,
                                file_name=model_filename,
                                mime="application/octet-stream",
                                help="点击下载训练好的模型文件，可用于后续预测"
                            )
                            
                            # 添加模型保存提示
                            st.success("模型已准备就绪，可下载保存。建议将模型文件保存在安全的位置。")
                            
                        except Exception as e:
                            st.error(f"模型保存过程中出现错误: {str(e)}")
                            
                        # 添加模型使用说明
                        with st.expander("📖 模型使用说明"):
                            st.markdown("""
                            ### 如何使用保存的模型
                            
                            1. **加载模型**:
                            ```python
                            import joblib
                            model = joblib.load('your_model_filename.joblib')
                            ```
                            
                            2. **进行预测**:
                            ```python
                            # 准备输入数据 (与训练时相同的特征顺序)
                            input_data = [[value1, value2, ...]]
                            prediction = model.predict(input_data)
                            ```
                            
                            3. **注意事项**:
                            - 确保使用与训练时相同的数据预处理步骤
                            - 输入数据的特征顺序必须与训练时一致
                            - 分类特征需要使用相同的编码方式
                            
                            ### 模型文件内容
                            - 训练好的机器学习模型
                            - 数据预处理管道（标准化、编码等）
                            - 特征名称和类型信息
                            """)
                    else:
                        st.warning("请先训练模型，然后才能保存。")

# 页面3: 参数预测
elif page == "参数预测":
    st.header("🔮 参数预测")

    # ========== 新增：上传训练好的单模型（joblib/pkl） ==========
    use_uploaded_model = False
    uploaded_model = st.file_uploader("上传训练好的模型（joblib格式）", type=['pkl', 'joblib'], key="pred_model_upload")
    if uploaded_model is not None:
        model_loaded, target_loaded, features_loaded, ftypes_loaded = _load_model_from_joblib(uploaded_model)
        # 若对象里携带 target，更新会话 target
        if target_loaded is not None:
            st.session_state.target = target_loaded
        # 若携带 features/feature_types，则也存起来供本页控件使用
        if features_loaded:
            st.session_state.features = features_loaded
        if ftypes_loaded:
            st.session_state.feature_types = ftypes_loaded

        st.success(f"模型加载完成，目标变量：{st.session_state.get('target', '未知')}")
        st.session_state.model = model_loaded
        use_uploaded_model = True

    # ========== 兼容：若没上传，则使用会话内训练好的模型 ==========
    if not use_uploaded_model:
        if not st.session_state.trained or st.session_state.model is None:
            st.warning("请先上传模型，或在「模型训练」页面训练模型后再来预测。")
            st.stop()

    model = st.session_state.model
    features = st.session_state.features
    feature_types = st.session_state.feature_types

    st.subheader("输入参数进行预测")
    input_data = {}
    cols = st.columns(2)

    # 若有基片等分类特征，可以像你原来一样处理；下面是通用处理：
    for i, feature in enumerate(features):
        with cols[i % 2]:
            if feature_types.get(feature) == "数值型":
                data_min = float(st.session_state.data[feature].min())
                data_max = float(st.session_state.data[feature].max())
                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=data_min, max_value=data_max,
                    value=(data_min + data_max) / 2, step=1.0,
                    key=f"input_{feature}"
                )
            else:
                options = st.session_state.data[feature].unique().tolist()
                input_data[feature] = st.selectbox(
                    f"{feature}", options, key=f"input_{feature}"
                )

    input_df = pd.DataFrame([input_data])

    if st.button("预测"):
        pred = model.predict(input_df)
        pred_val = float(np.ravel(pred)[0])
        st.success(f"🔬 预测的 **{st.session_state.target}** 为：{pred_val:.4f}")
        st.subheader("输入参数详情")
        st.dataframe(input_df)
        
        # 保存预测结果到session state
        st.session_state.last_prediction = pred_val
        st.session_state.last_input_data = input_data
    
    # ... (保持原有代码不变)
        # 新增部分：CrSb薄膜生长结束图像展示（基于实际结构的示意图）
        st.markdown("---")
        st.subheader("CrSb薄膜原子结构示意图")
        
        # 使用两列布局
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CrSb晶体结构示意图**")
            
            # 创建CrSb晶体结构示意图
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # CrSb具有NiAs型结构（六方晶系）
            # 晶格常数: a = 4.12 Å, c = 5.42 Å
            a = 1.0  # 相对晶格常数a
            c = 1.3  # 相对晶格常数c (c/a ≈ 1.315，符合CrSb的实际比例)
            
            # 获取化学计量比（从预测数据或输入数据）
            stoichiometry = 1.0  # 默认1:1
            
            # 检查是否有化学计量比相关的参数
            stoichiometry_params = [f for f in features if "ratio" in f.lower() or "化学计量" in f or "stoichiometry" in f.lower()]
            
            # 如果已进行预测，尝试从预测数据获取化学计量比
            if hasattr(st.session_state, 'last_input_data'):
                # 如果目标变量是化学计量比，使用预测值
                if any(keyword in st.session_state.target.lower() for keyword in ["ratio", "化学计量", "stoichiometry"]):
                    stoichiometry = st.session_state.last_prediction
                # 否则从输入特征中寻找化学计量比参数
                elif stoichiometry_params:
                    # 使用第一个找到的化学计量比参数
                    stoichiometry_param = stoichiometry_params[0]
                    stoichiometry = st.session_state.last_input_data.get(stoichiometry_param, 1.0)
            
            # 确保stoichiometry不为零，避免除以零错误
            if stoichiometry <= 0:
                stoichiometry = 1.0
                
            # 计算Cr和Sb的比例
            cr_ratio = 1.0 / (1.0 + stoichiometry) if stoichiometry != 1.0 else 0.5
            sb_ratio = stoichiometry / (1.0 + stoichiometry) if stoichiometry != 1.0 else 0.5
            
            # 绘制六方晶格结构 - 更符合实际的CrSb结构
            # 在NiAs型结构中，Cr和Sb原子交替排列形成六方最密堆积
            
            # 绘制基片底层
            ax.add_patch(plt.Rectangle((-0.5, -0.5), 4, 0.2, color='lightgray', alpha=0.7, label='基片'))
            
            # 绘制多个晶胞以显示完整的晶体结构
            for i in range(3):
                for j in range(3):
                    # 计算晶胞原点
                    origin_x = i * a
                    origin_y = j * c
                    
                    # 绘制Cr原子（红色）- 位于六方晶格的顶点和体心
                    ax.scatter(origin_x, origin_y, color='red', s=100, label='Cr' if i == 0 and j == 0 else "")
                    ax.scatter(origin_x + a/2, origin_y + c/2, color='red', s=100)
                    
                    # 绘制Sb原子（蓝色）- 位于六方晶格的面心位置
                    ax.scatter(origin_x + a/2, origin_y, color='blue', s=100, label='Sb' if i == 0 and j == 0 else "")
                    ax.scatter(origin_x, origin_y + c/2, color='blue', s=100)
                    
                    # 添加晶胞边界
                    if i < 2 and j < 2:  # 只绘制内部晶胞边界
                        ax.plot([origin_x, origin_x + a], [origin_y, origin_y], 'k-', alpha=0.3)
                        ax.plot([origin_x, origin_x], [origin_y, origin_y + c], 'k-', alpha=0.3)
                        ax.plot([origin_x + a, origin_x + a], [origin_y, origin_y + c], 'k-', alpha=0.3)
                        ax.plot([origin_x, origin_x + a], [origin_y + c, origin_y + c], 'k-', alpha=0.3)
            
            # 添加晶格方向指示
            ax.arrow(0, 0, a, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
            ax.text(a/2, -0.2, '[100]', ha='center')
            
            ax.arrow(0, 0, 0, c, head_width=0.1, head_length=0.1, fc='k', ec='k')
            ax.text(-0.2, c/2, '[001]', ha='center', va='center')
            
            ax.set_xlim(-0.5, 3.0)
            ax.set_ylim(-0.5, 3.5)
            ax.set_xlabel('晶格方向 [100]')
            ax.set_ylabel('晶格方向 [001]')
            ax.set_title(f'CrSb六方晶体结构 (NiAs型)')
            ax.grid(False)
            ax.legend()
            
            st.pyplot(fig)
            
            st.markdown(f"""
            **晶体结构特征:**
            - 空间群: P6₃/mmc (No. 194)
            - 晶格常数: a = 4.12 Å, c = 5.42 Å
            - 结构类型: NiAs型六方结构
            - 原子排列: Cr(红色)和Sb(蓝色)原子交替排列形成六方最密堆积
            - 化学计量比: Cr:Sb ≈ {1/stoichiometry:.2f}:{stoichiometry:.2f} (预测值)
            """)
       
            
        with col2:
            st.markdown("**CrSb薄膜生长示意图**")
        
            # 创建薄膜生长示意图
            fig, ax = plt.subplots(figsize=(6, 6))
            
            # 绘制基片
            ax.add_patch(plt.Rectangle((-0.25, 0), 4, 0.5, color='gray', alpha=0.7, label='基片'))
            
            # 获取化学计量比（如果有）
            stoichiometry = pred_val if 'prediction' in locals() else 1.0  # 使用预测值或默认1:1
            
            # 检查是否有化学计量比相关的参数
            if stoichiometry_params and hasattr(st.session_state, 'last_input_data'):
                # 使用第一个找到的化学计量比参数
                stoichiometry_param = stoichiometry_params[0]
                stoichiometry = st.session_state.last_input_data.get(stoichiometry_param, 1.0)
            
            # 计算Cr和Sb的比例
            cr_ratio = 1.0 / (1.0 + stoichiometry) if stoichiometry != 1.0 else 0.5
            sb_ratio = stoichiometry / (1.0 + stoichiometry) if stoichiometry != 1.0 else 0.5
            
           # 绘制薄膜层（多层原子）- 随机分布缺失原子
        # 绘制薄膜层（多层原子）- 交替放置 + 缺失随机分布
            layers = 5
            atoms_per_layer = 6
            total_atoms = layers * atoms_per_layer

            # 计算每种原子的总数
            cr_total_needed = (total_atoms + 1) // 2  # 偶数格子位置
            sb_total_needed = total_atoms // 2        # 奇数格子位置

            # 按比例决定实际能放的数量
            cr_count = min(cr_total_needed, int(total_atoms * cr_ratio))
            sb_count = min(sb_total_needed, int(total_atoms * sb_ratio))

            # 确定哪些位置是 Cr 位、哪些是 Sb 位
            cr_positions = [i for i in range(total_atoms) if i % 2 == 0]
            sb_positions = [i for i in range(total_atoms) if i % 2 == 1]

            # 随机挑选缺失位置
            missing_cr = set(np.random.choice(cr_positions, cr_total_needed - cr_count, replace=False)) \
                if cr_total_needed > cr_count else set()
            missing_sb = set(np.random.choice(sb_positions, sb_total_needed - sb_count, replace=False)) \
                if sb_total_needed > sb_count else set()

            atom_index = 0
            for layer in range(layers):
                y_pos = 0.5 + layer * 0.4
                for i in range(atoms_per_layer):
                    x_pos = i * 0.7

                    if atom_index % 2 == 0:  # Cr 位置
                        if atom_index in missing_cr:
                            ax.scatter(x_pos, y_pos, facecolors='none', edgecolors='black', s=80,
                                    linewidths=1.5, label='Cr缺失' if atom_index == list(missing_cr)[0] else "")
                        else:
                            ax.scatter(x_pos, y_pos, color='red', s=80,
                                    label='Cr原子' if atom_index == 0 else "")
                    else:  # Sb 位置
                        if atom_index in missing_sb:
                            ax.scatter(x_pos, y_pos, facecolors='none', edgecolors='black', s=80,
                                    linewidths=1.5, label='Sb缺失' if atom_index == list(missing_sb)[0] else "")
                        else:
                            ax.scatter(x_pos, y_pos, color='blue', s=80,
                                    label='Sb原子' if atom_index == 1 else "")

                    atom_index += 1


            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(0, 3.5)
            ax.set_xlabel('横向位置')
            ax.set_ylabel('生长方向')
            ax.set_title(f'CrSb薄膜生长示意图 (Cr:Sb ≈ {1}:{stoichiometry:.2f})')
            ax.legend()
            
            st.pyplot(fig)
            
            # 根据化学计量比评估薄膜质量
            if abs(stoichiometry - 1.0) < 0.1:
                quality = "高质量"
                quality_desc = "接近理想化学计量比，晶体结构完整，缺陷少"
            elif abs(stoichiometry - 1.0) < 0.3:
                quality = "中等质量"
                quality_desc = "化学计量比略有偏离，可能存在少量缺陷"
            else:
                quality = "低质量"
                quality_desc = "化学计量比严重偏离，可能存在大量缺陷和非晶区域"
            
            st.markdown(f"""
            **生长特征:**
            - 化学计量比: Cr:Sb ≈ {1}:{stoichiometry:.2f} (预测值)
            - 薄膜质量: {quality}
            - 质量描述: {quality_desc}
            - 生长模式: 层状外延生长
            - 原子分布: {cr_count}个Cr原子, {sb_count}个Sb原子
            """)
        # 添加CrSb薄膜特性说明
        st.markdown("""
        ### CrSb薄膜特性说明
        
        CrSb是一种重要的半金属磁性材料，具有以下特性:
        
        - **晶体结构**: 六方晶系，NiAs型结构，空间群P6₃/mmc
        - **磁性**: 室温铁磁性，居里温度约700K
        - **电子结构**: 半金属性，自旋极化率接近100%
        - **应用**: 自旋电子器件、磁存储设备、磁传感器
        
        **化学计量比对薄膜质量的影响:**
        - 理想化学计量比 (Cr:Sb = 1:1): 高质量晶体，磁性能最佳
        - Cr过量 (Cr:Sb > 1:1): 可能形成Cr团簇，降低磁性能
        - Sb过量 (Cr:Sb < 1:1): 可能导致非晶区域，降低结晶质量
        
        **高质量CrSb薄膜的生长关键参数:**
        - 基片温度: 250-400°C
        - 沉积速率: 0.1-0.5 Å/s
        - 溅射功率: 50-100W (DC溅射)
        - 氩气压力: 2-5 mTorr
        - 基片类型: MgO、Al₂O₃或Si with buffer layer
        """)
# 页面3-2：反向预测（多目标）
elif page == "反向预测":
    st.header("🔄 反向参数推荐（多目标优化）")

    df_data = st.session_state.get("data", None)
    if df_data is None or df_data.empty:
        st.warning("请先在前面的页面上传数据集，并完成特征/目标设置。")
        st.stop()

    # ====== 多模型上传：一个 target 对应一个模型（可多选上传）======
    st.subheader("上传一个或多个已训练模型（每个文件一个目标变量）")
    uploaded_models = st.file_uploader(
        "支持 joblib/pkl，可一次选择多个文件",
        type=["pkl", "joblib"],
        accept_multiple_files=True,
        key="rev_models_upload"
    )

    models_by_target = {}   # {target: {"model":..., "features":..., "feature_types":...}}

    if uploaded_models:
        for f in uploaded_models:
            m, t, feats, ftypes = _load_model_from_joblib(f)
            # 若缺失 target，则允许用户补充命名
            if t is None:
                t = st.text_input(f"为文件 {f.name} 指定该模型的目标变量名", value=f"target_{len(models_by_target)+1}", key=f"tname_{f.name}")
            st.write(f"✔️ 模型 {f.name} 加载完成，目标变量：**{t}**")
            models_by_target[t] = {"model": m, "features": feats, "feature_types": ftypes}

    # 兜底：若未上传任何模型，则尝试使用会话中训练好的单模型
    if not models_by_target:
        if st.session_state.get("trained", False) and st.session_state.get("model", None) is not None:
            t = st.session_state.get("target", "target")
            models_by_target[t] = {
                "model": st.session_state.model,
                "features": st.session_state.get("features", []),
                "feature_types": st.session_state.get("feature_types", {})
            }
            st.info(f"未上传外部模型，使用当前会话模型（目标变量：{t}）。")
        else:
            st.warning("请至少上传一个模型，或先在会话中训练一个模型。")
            st.stop()

    # ====== 选择要同时优化的目标 ======
    all_targets_loaded = list(models_by_target.keys())
    target_options = st.multiselect(
        "选择需要同时优化的薄膜特性（目标变量）",
        options=all_targets_loaded,
        default=all_targets_loaded
    )
    if len(target_options) == 0:
        st.warning("请至少选择一个目标特性")
        st.stop()

    # ====== 统一特征&类型（用于构造候选样本）。若不统一，则按候选里“并集”处理 ======
    # 若会话里已有 features/feature_types，优先使用；否则根据模型/数据推断
    global_features = st.session_state.get("features", [])
    global_ftypes = st.session_state.get("feature_types", {})
    if not global_features:
        # 从任意一个模型中推断
        for t in target_options:
            global_features = models_by_target[t].get("features", [])
            if not global_features:
                global_features = _infer_features_if_missing(models_by_target[t]["model"], df_data, known_targets=target_options)
            if global_features:
                break
    # 若还为空，就用数据列减去目标列
    if not global_features:
        global_features = [c for c in df_data.columns if c not in target_options]
    if not global_ftypes:
        global_ftypes = _infer_feature_types_if_missing(global_features, df_data, given=None)

    # ====== 参数范围/可选值限制（数值：范围；分类型：多选）======
    st.markdown("### 限定生长参数范围或固定值")
    param_limits = {}
    cols_lim = st.columns(2)
    for i, f in enumerate(global_features):
        with cols_lim[i % 2]:
            if global_ftypes.get(f) == "数值型":
                # 给出稍拓的范围（±10%）
                dmin = float(df_data[f].min())
                dmax = float(df_data[f].max())
                lo, hi = 0.9 * dmin, 1.1 * dmax
                # 滑条范围
                slider_vals = st.slider(
                    f"{f} 范围",
                    min_value=float(min(lo, dmin)), max_value=float(max(hi, dmax)),
                    value=(float(dmin), float(dmax)),
                    step=1.0, key=f"rev_rng_{f}"
                )
                # 数字输入精修
                c1, c2 = st.columns(2)
                with c1:
                    low_val = st.number_input(f"{f} 最小值", value=float(slider_vals[0]), step=1.0, key=f"rev_min_{f}")
                with c2:
                    high_val = st.number_input(f"{f} 最大值", value=float(slider_vals[1]), step=1.0, key=f"rev_max_{f}")
                if low_val > high_val:
                    low_val, high_val = high_val, low_val
                    st.warning(f"已自动调整 {f} 的范围以保证最小值 ≤ 最大值")
                param_limits[f] = (float(low_val), float(high_val))
            else:
                opts = df_data[f].dropna().unique().tolist()
                selected = st.multiselect(f"{f} 可选值", options=opts, default=opts, key=f"rev_cat_{f}")
                if len(selected) == 0:
                    st.warning(f"{f} 当前无可选值，已临时使用全部历史值作为候选。")
                    selected = opts
                param_limits[f] = selected

    # ====== 设置目标值和权重 ======
    st.markdown("### 设置目标薄膜特性期望值（可加权）")
    target_vals, target_weights = {}, {}
    tcols = st.columns(2)
    for i, t in enumerate(target_options):
        with tcols[i % 2]:
            default_val = float(df_data[t].mean()) if t in df_data.columns else 0.0
            val = st.number_input(f"{t} 目标值", value=default_val, key=f"rev_tval_{t}")
            w = st.number_input(f"{t} 权重", value=1.0, min_value=0.0, step=0.1, key=f"rev_tw_{t}")
            target_vals[t] = float(val)
            target_weights[t] = float(w)

    # ====== 采样规模与输出设置 ======
    st.markdown("### 生成与筛选设置")
    colA, colB, colC = st.columns(3)
    with colA:
        n_samples = st.number_input("候选采样数量", min_value=200, max_value=20000, value=2000, step=200, key="rev_nsamp")
    with colB:
        top_k = st.number_input("显示前 K 组", min_value=5, max_value=100, value=10, step=5, key="rev_topk")
    with colC:
        round_int = st.checkbox("结果中数值特征四舍五入为整数", value=True, key="rev_roundint")

    if st.button("生成推荐参数组合"):
        # 1) 候选采样
        candidates_df = _sample_candidates(int(n_samples), global_features, global_ftypes, param_limits)

        # 对于分类型里出现 None（因为用户把可选值清空）做兜底填充
        for f in global_features:
            if global_ftypes.get(f) == "分类型":
                if candidates_df[f].isna().any():
                    fallback = df_data[f].dropna().unique().tolist()
                    if fallback:
                        candidates_df.loc[candidates_df[f].isna(), f] = np.random.choice(fallback, size=candidates_df[f].isna().sum())

        # 2) 逐 target 预测（多模型）
        preds_df = _predict_with_models(candidates_df, {t: models_by_target[t] for t in target_options}, global_features)

        # 3) 误差计算（加权 L1）
        err = _compute_weighted_error(preds_df, target_vals, target_weights)
        candidates_df = candidates_df.reset_index(drop=True)
        result_df = pd.concat([candidates_df, preds_df], axis=1)
        result_df["weighted_error"] = err

        # 4) 选出前 K 组
        best_df = result_df.sort_values("weighted_error", ascending=True).head(int(top_k)).copy()

        # 按需把数值型特征取整
        if round_int:
            for f in global_features:
                if global_ftypes.get(f) == "数值型":
                    best_df[f] = best_df[f].round(0).astype(int)

        # 展示 & 下载
        st.subheader("推荐的参数组合（加权误差最小的前K组）")
        st.dataframe(best_df.drop(columns=["weighted_error"]))

        csv_bytes = best_df.to_csv(index=False).encode("utf-8")
        st.download_button("下载推荐结果 CSV", data=csv_bytes, file_name="reverse_recommendations.csv", mime="text/csv")

# 页面4: 特征分析
# 修改特征分析页面的特征重要性展示部分
elif page == "特征分析":
    st.header("📈 特征重要性分析")
    
   # 上传训练好的模型（joblib）
    uploaded_model = st.file_uploader("上传训练好的模型（joblib格式）", type=['pkl', 'joblib'], key="stab_model_upload")
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("模型加载完成")
        # 如果上传了模型，使用上传的模型
        use_uploaded_model = True
    elif hasattr(st.session_state, 'model') and st.session_state.trained:
        model = st.session_state.model
        st.info("使用当前会话训练的模型")
        use_uploaded_model = False
    else:
        st.warning("请先训练模型或上传已训练模型")
        st.stop()
    
    if not st.session_state.trained and not use_uploaded_model:
        st.warning("请先在「模型训练」页面训练模型或上传一个训练好的模型")
    else:
        if use_uploaded_model:
            model = model
            # 尝试从上传的模型中获取模型类型
            model_type = "上传的模型"
        else:
            model = st.session_state.model
            model_type = st.session_state.get('model_type', '未知模型')
        
        # 获取特征重要性
        regressor = model.named_steps['regressor']
        preprocessor = model.named_steps['preprocessor']
        
        # 获取特征名称
        numeric_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "数值型"]
        categorical_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "分类型"]
        
        # 如果基片特征在分类特征中，确保它不会被重复处理
        if hasattr(st.session_state, 'substrate_feature') and st.session_state.substrate_feature in categorical_features:
            categorical_features = [f for f in categorical_features if f != st.session_state.substrate_feature]
        
        # 获取分类特征编码后的名称
        if categorical_features:
            encoder = preprocessor.named_transformers_['cat']
            encoded_names = encoder.get_feature_names_out(categorical_features)
        else:
            encoded_names = []
        
        feature_names = numeric_features + list(encoded_names)
        
        # 根据模型类型获取特征重要性
        st.subheader(f"特征重要性分析 ({model_type})")
        if hasattr(regressor, 'feature_importances_'):
            importances = regressor.feature_importances_
            importance_type = "重要性得分"
        elif hasattr(regressor, 'coef_'):
            # 对于线性模型，使用系数的绝对值作为重要性
            importances = np.abs(regressor.coef_)
            importance_type = "系数绝对值 (表示影响强度)"
        else:
            st.warning("此模型类型不支持特征重要性分析")
            st.stop()
        
        # 创建特征重要性数据框
        importance_df = pd.DataFrame({
            '特征': feature_names,
            importance_type: importances
        }).sort_values(importance_type, ascending=False)
        
        # 绘制特征重要性图表
        st.subheader(f"特征{importance_type}排序")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importance_type, y='特征', data=importance_df, ax=ax)
        ax.set_title(f'特征{importance_type}')
        st.pyplot(fig)
        
        # 显示特征重要性表格
        st.dataframe(importance_df)
        
        # 其余部分保持不变...
        # 部分依赖图（简化版）
        st.subheader("参数影响分析")
        st.write("选择特征查看其与目标变量的关系")
        
        # 只显示数值型特征
        numeric_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "数值型"]
        selected_feature = st.selectbox("选择特征", numeric_features)
        
        if selected_feature:
            # 创建部分依赖图数据
            feature_data = st.session_state.data[selected_feature]
            grid = np.linspace(feature_data.min(), feature_data.max(), 50)

            # 只对数值型特征求中位数，分类型特征用众数
            numeric_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "数值型"]
            categorical_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "分类型"]

            baseline_data = []
            for val in grid:
                sample = {}
                # 数值型特征用中位数
                for f in numeric_features:
                    sample[f] = st.session_state.data[f].median()
                # 分类型特征用众数
                for f in categorical_features:
                    sample[f] = st.session_state.data[f].mode()[0]
                # 当前特征用遍历值
                sample[selected_feature] = val
                baseline_data.append(sample)

            baseline_df = pd.DataFrame(baseline_data)

            # 预测
            predictions = model.predict(baseline_df)

            # 绘制部分依赖图
            fig, ax = plt.subplots()
            ax.plot(grid, predictions, 'b-')
            ax.set_xlabel(selected_feature)
            ax.set_ylabel(f"预测 {st.session_state.target}")
            ax.set_title(f"{selected_feature} 对 {st.session_state.target} 的影响")
            st.pyplot(fig)
                    # 新增部分：双参数关系分析
            st.subheader("双参数关系分析")
            st.write("选择两个数值型参数观察它们的关系，并通过颜色表示目标变量")
            numeric_features = [f for f in st.session_state.features if st.session_state.feature_types[f] == "数值型"]
            # 确保有足够的数值型特征
            if len(numeric_features) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_feature = st.selectbox("选择X轴参数", numeric_features, key="x_feature")
                with col2:
                    # 默认选择不同于X轴的参数
                    y_options = [f for f in numeric_features if f != x_feature]
                    default_idx = 0 if len(y_options) > 0 else -1
                    if default_idx >= 0:
                        y_feature = st.selectbox("选择Y轴参数", y_options, key="y_feature", index=default_idx)
                    else:
                        y_feature = st.selectbox("选择Y轴参数", numeric_features, key="y_feature")
                
                # 添加数据源选择复选框
                st.write("选择要显示的数据源:")
                col_show1, col_show2, col_show3 = st.columns(3)
                with col_show1:
                    show_original = st.checkbox("原始数据", value=True)
                with col_show2:
                    show_augmented = st.checkbox("增强数据", value=True)
                with col_show3:
                    show_predicted = st.checkbox("预测值", value=True)
                
                if x_feature and y_feature and x_feature != y_feature:
                    # 创建散点图
                    fig, ax = plt.subplots(figsize=(10, 8))
                    scatter_handles = []
                    scatter_labels = []
                    
                    # 1. 绘制原始数据
                    if show_original and st.session_state.data is not None:
                        x_original = st.session_state.data[x_feature]
                        y_original = st.session_state.data[y_feature]
                        target_original = st.session_state.data[st.session_state.target]
                        original_scatter = ax.scatter(
                            x_original, y_original, 
                            c=target_original, cmap='viridis', 
                            alpha=0.6, marker='o', s=50
                        )
                        scatter_handles.append(original_scatter)
                        scatter_labels.append('原始数据')
                    
                    # 2. 绘制增强数据（如果有）
                    if show_augmented and hasattr(st.session_state, 'X_train_augmented'):
                        x_augmented = st.session_state.X_train_augmented[x_feature]
                        y_augmented = st.session_state.X_train_augmented[y_feature]
                        if hasattr(st.session_state, 'y_train_augmented'):
                            target_augmented = st.session_state.y_train_augmented
                            augmented_scatter = ax.scatter(
                                x_augmented, y_augmented, 
                                c=target_augmented, cmap='plasma', 
                                alpha=0.5, marker='^', s=40
                            )
                            scatter_handles.append(augmented_scatter)
                            scatter_labels.append('增强数据')
                    
                    # 3. 绘制预测值（如果有）
                    if show_predicted and hasattr(st.session_state, 'X_test') and hasattr(st.session_state, 'y_pred'):
                        x_pred = st.session_state.X_test[x_feature]
                        y_pred = st.session_state.X_test[y_feature]
                        target_pred = st.session_state.y_pred
                        pred_scatter = ax.scatter(
                            x_pred, y_pred, 
                            c=target_pred, cmap='cividis', 
                            alpha=0.7, marker='s', s=60
                        )
                        scatter_handles.append(pred_scatter)
                        scatter_labels.append('预测值')
                    
                    # 添加颜色条和图例
                    if scatter_handles:  # 确保有图形元素
                        cbar = plt.colorbar(scatter_handles[0])
                        cbar.set_label(st.session_state.target)
                        ax.legend(scatter_handles, scatter_labels, loc='best')
                    
                    ax.set_xlabel(x_feature)
                    ax.set_ylabel(y_feature)
                    ax.set_title(f"{x_feature} 与 {y_feature} 的关系\n(颜色表示 {st.session_state.target})")
                    
                    # 添加网格
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
                    
                    # 计算原始数据相关系数
                    if st.session_state.data is not None:
                        corr = np.corrcoef(st.session_state.data[x_feature], st.session_state.data[y_feature])[0, 1]
                        st.write(f"**{x_feature}** 与 **{y_feature}** 的相关系数: {corr:.3f}")
                        
                        # 解释相关性
                        if abs(corr) < 0.2:
                            strength = "弱"
                        elif abs(corr) < 0.6:
                            strength = "中等"
                        else:
                            strength = "强"
                        
                        direction = "正" if corr > 0 else "负"
                        st.write(f"**相关性解释**: {strength}{direction}相关")
                elif x_feature == y_feature:
                    st.warning("请选择两个不同的参数进行分析")
            else:
                st.warning("需要至少两个数值型参数才能进行双参数关系分析")
# 页面5: 参数分布分析
elif page == "参数分布分析":
    st.header("📊 参数分布分析")
    
    # 上传训练好的模型（joblib）
    uploaded_model = st.file_uploader("上传训练好的模型（joblib格式）", type=['pkl', 'joblib'], key="dist_model_upload")
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("模型加载完成")
        # 如果上传了模型，使用上传的模型
        use_uploaded_model = True
    elif hasattr(st.session_state, 'model') and st.session_state.trained:
        model = st.session_state.model
        st.info("使用当前会话训练的模型")
        use_uploaded_model = False
    else:
        st.warning("请先训练模型或上传已训练模型")
        st.stop()
    
    if st.session_state.data is None:
        st.warning("请先在「数据上传与设置」页面上传数据")
    else:
        df = st.session_state.data
        target = st.session_state.target
        features = st.session_state.features
        feature_types = st.session_state.feature_types
        
        st.subheader("参数对观测量的分布影响分析")
        
        # 选择分析参数
        selected_feature = st.selectbox("选择要分析的参数", features)
        
        if selected_feature:
            # 获取参数类型
            feature_type = feature_types[selected_feature]
            
            # 创建两列布局
            col1, col2 = st.columns(2)
            
            with col1:
                # 散点图（适用于数值型参数）
                if feature_type == "数值型":
                    # 确保数据是数值型
                    try:
                        feature_data = pd.to_numeric(df[selected_feature], errors='coerce')
                        target_data = pd.to_numeric(df[target], errors='coerce')
                        
                        # 移除NaN值
                        valid_data = pd.notna(feature_data) & pd.notna(target_data)
                        feature_data = feature_data[valid_data]
                        target_data = target_data[valid_data]
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.scatter(feature_data, target_data, alpha=0.6)
                        ax.set_xlabel(selected_feature)
                        ax.set_ylabel(target)
                        ax.set_title(f"{selected_feature} 与 {target} 的关系")
                        
                        # 添加趋势线
                        if len(feature_data) > 1:
                            z = np.polyfit(feature_data, target_data, 1)
                            p = np.poly1d(z)
                            ax.plot(feature_data, p(feature_data), "r--", alpha=0.8)
                            
                            # 计算相关系数
                            corr = np.corrcoef(feature_data, target_data)[0, 1]
                            ax.text(0.05, 0.95, f'相关系数: {corr:.3f}', transform=ax.transAxes, 
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"创建散点图时出错: {str(e)}")
                
                # 箱线图（适用于分类型参数）
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        categories = df[selected_feature].unique()
                        
                        # 为每个类别创建数据列表
                        data = []
                        labels = []
                        for category in categories:
                            category_data = df[df[selected_feature] == category][target]
                            # 尝试转换为数值型
                            try:
                                category_data = pd.to_numeric(category_data, errors='coerce').dropna()
                                if len(category_data) > 0:  # 确保有数据
                                    data.append(category_data)
                                    labels.append(str(category))
                            except:
                                # 如果无法转换为数值型，跳过
                                continue
                        
                        if data:  # 确保有有效数据
                            ax.boxplot(data, labels=labels)
                            ax.set_xlabel(selected_feature)
                            ax.set_ylabel(target)
                            ax.set_title(f"{selected_feature} 对 {target} 的分布影响")
                            plt.xticks(rotation=45)
                            st.pyplot(fig)
                        else:
                            st.warning("没有有效的数值数据可显示")
                    except Exception as e:
                        st.error(f"创建箱线图时出错: {str(e)}")
            
            with col2:
                # 分布直方图（适用于数值型参数）
                if feature_type == "数值型":
                    try:
                        # 确保数据是数值型
                        feature_data = pd.to_numeric(df[selected_feature], errors='coerce').dropna()
                        target_data = pd.to_numeric(df[target], errors='coerce').dropna()
                        
                        # 选择区间数
                        bins = st.slider("直方图区间数", 5, 50, 20)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # 计算参数的分位数
                        q25, q50, q75 = np.percentile(feature_data, [25, 50, 75])
                        
                        # 创建子区间
                        low_range = feature_data <= q25
                        mid_range = (feature_data > q25) & (feature_data <= q75)
                        high_range = feature_data > q75
                        
                        # 绘制分布
                        ax.hist(target_data[low_range], bins=bins, alpha=0.5, label=f"低 {selected_feature} (≤{q25:.2f})")
                        ax.hist(target_data[mid_range], bins=bins, alpha=0.5, label=f"中 {selected_feature} ({q25:.2f}-{q75:.2f})")
                        ax.hist(target_data[high_range], bins=bins, alpha=0.5, label=f"高 {selected_feature} (>{q75:.2f})")
                        
                        ax.set_xlabel(target)
                        ax.set_ylabel("频数")
                        ax.set_title(f"不同 {selected_feature} 区间的 {target} 分布")
                        ax.legend()
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"创建直方图时出错: {str(e)}")
                
                # 小提琴图（适用于分类型参数）
                else:
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        categories = df[selected_feature].unique()
                        
                        # 为每个类别创建数据列表
                        data = []
                        labels = []
                        for category in categories:
                            category_data = df[df[selected_feature] == category][target]
                            # 尝试转换为数值型
                            try:
                                category_data = pd.to_numeric(category_data, errors='coerce').dropna()
                                if len(category_data) > 0:  # 确保有数据
                                    data.append(category_data)
                                    labels.append(str(category))
                            except:
                                # 如果无法转换为数值型，跳过
                                continue
                        
                        if data:  # 确保有有效数据
                            # 创建小提琴图
                            parts = ax.violinplot(data, showmeans=True, showmedians=True)
                            
                            # 设置颜色
                            for pc in parts['bodies']:
                                pc.set_facecolor('lightblue')
                                pc.set_alpha(0.6)
                            
                            ax.set_xlabel(selected_feature)
                            ax.set_ylabel(target)
                            ax.set_title(f"{selected_feature} 对 {target} 的分布影响")
                            ax.set_xticks(range(1, len(labels) + 1))
                            ax.set_xticklabels(labels, rotation=45)
                            st.pyplot(fig)
                        else:
                            st.warning("没有有效的数值数据可显示")
                    except Exception as e:
                        st.error(f"创建小提琴图时出错: {str(e)}")
            
            # 统计分析
            st.subheader("统计分析")
            
            if feature_type == "数值型":
                try:
                    # 确保数据是数值型
                    feature_data = pd.to_numeric(df[selected_feature], errors='coerce').dropna()
                    target_data = pd.to_numeric(df[target], errors='coerce').dropna()
                    
                    # 计算相关系数和p值
                    if len(feature_data) > 1 and len(target_data) > 1:
                        corr, p_value = stats.pearsonr(feature_data, target_data)
                        st.write(f"**{selected_feature}** 与 **{target}** 的 Pearson 相关系数: {corr:.3f} (p值: {p_value:.3e})")
                        
                        # 解释相关性强度
                        if abs(corr) < 0.3:
                            strength = "弱相关"
                        elif abs(corr) < 0.7:
                            strength = "中等相关"
                        else:
                            strength = "强相关"
                        
                        direction = "正" if corr > 0 else "负"
                        st.write(f"**相关性解释**: {strength}{direction}相关")
                    else:
                        st.warning("数据不足，无法计算相关性")
                except Exception as e:
                    st.error(f"计算相关性时出错: {str(e)}")
            
            else:
                try:
                    # 对分类变量进行ANOVA分析
                    categories = df[selected_feature].unique()
                    group_data = []
                    valid_categories = []
                    
                    for category in categories:
                        category_data = df[df[selected_feature] == category][target]
                        # 尝试转换为数值型
                        try:
                            category_data = pd.to_numeric(category_data, errors='coerce').dropna()
                            if len(category_data) > 0:  # 确保有数据
                                group_data.append(category_data)
                                valid_categories.append(category)
                        except:
                            # 如果无法转换为数值型，跳过
                            continue
                    
                    if len(group_data) >= 2:
                        f_stat, p_value = stats.f_oneway(*group_data)
                        st.write(f"ANOVA 分析 F统计量: {f_stat:.3f} (p值: {p_value:.3e})")
                        
                        if p_value < 0.05:
                            st.write("**统计显著性**: 不同组别之间存在显著差异 (p < 0.05)")
                        else:
                            st.write("**统计显著性**: 不同组别之间没有显著差异 (p ≥ 0.05)")
                    else:
                        st.warning("数据不足，无法进行ANOVA分析")
                    
                    # 显示各组的描述性统计
                    st.write("**各组描述性统计**:")
                    desc_stats = []
                    
                    for category in valid_categories:
                        category_data = df[df[selected_feature] == category][target]
                        # 尝试转换为数值型
                        try:
                            category_data = pd.to_numeric(category_data, errors='coerce').dropna()
                            if len(category_data) > 0:  # 确保有数据
                                desc_stats.append({
                                    '类别': str(category),
                                    '数量': len(category_data),
                                    '均值': category_data.mean(),
                                    '标准差': category_data.std(),
                                    '最小值': category_data.min(),
                                    '最大值': category_data.max()
                                })
                        except:
                            # 如果无法转换为数值型，跳过
                            continue
                    
                    if desc_stats:
                        desc_df = pd.DataFrame(desc_stats)
                        st.dataframe(desc_df.round(3))
                    else:
                        st.warning("没有有效的数值数据可显示")
                except Exception as e:
                    st.error(f"进行统计分析时出错: {str(e)}")

# 页面6: 模型稳定性验证
elif page == "模型稳定性验证":
    st.header("🔍 模型稳定性验证")
    
    # 上传训练好的模型（joblib）
    uploaded_model = st.file_uploader("上传训练好的模型（joblib格式）", type=['pkl', 'joblib'], key="stab_model_upload")
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)
        st.success("模型加载完成")
        # 如果上传了模型，使用上传的模型
        use_uploaded_model = True
    elif hasattr(st.session_state, 'model') and st.session_state.trained:
        model = st.session_state.model
        st.info("使用当前会话训练的模型")
        use_uploaded_model = False
    else:
        st.warning("请先训练模型或上传已训练模型")
        st.stop()
    
    st.info("""
    模型稳定性验证通过多种方法评估模型的可靠性:
    - **交叉验证**: 多次随机划分数据，评估模型性能的一致性
    - **自助法**: 通过重采样评估模型对数据变化的敏感性
    """)
    
    # 选择验证方法
    validation_method = st.selectbox(
        "选择稳定性验证方法",
        ["交叉验证", "自助法"]
    )
    
    if validation_method == "交叉验证":
        st.subheader("交叉验证稳定性分析")
        
        n_folds = st.slider("交叉验证折数", 3, 10, 5)
        n_repeats = st.slider("重复次数", 1, 10, 3)
        
        if st.button("开始交叉验证"):
            with st.spinner("正在进行交叉验证..."):
                from sklearn.model_selection import cross_val_score, RepeatedKFold
                
                # 获取数据和模型
                df = st.session_state.data
                features = st.session_state.features
                target = st.session_state.target
                feature_types = st.session_state.feature_types
                
                # 根据基片选择过滤数据
                if st.session_state.substrate_option == "特定基片" and st.session_state.selected_substrate:
                    substrate_feature = st.session_state.substrate_feature
                    df = df[df[substrate_feature] == st.session_state.selected_substrate]
                
                X = df[features]
                y = df[target]
                
                # 构建预处理管道
                numeric_features = [f for f in features if feature_types[f] == "数值型"]
                categorical_features = [f for f in features if feature_types[f] == "分类型"]
                
                if hasattr(st.session_state, 'substrate_feature') and st.session_state.substrate_feature in categorical_features:
                    categorical_features = [f for f in categorical_features if f != st.session_state.substrate_feature]
                
                preprocessor = ColumnTransformer([
                    ('num', StandardScaler(), numeric_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                ])
                
                # 获取模型类型和参数
                model_type = st.session_state.model_type
                model = st.session_state.model
                regressor = model.named_steps['regressor']
                
                # 创建完整管道
                full_model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', regressor)
                ])
                
                # 执行重复K折交叉验证
                cv = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=42)
                cv_scores = cross_val_score(full_model, X, y, cv=cv, scoring='r2')
                
                # 计算统计量
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                cv_range = np.ptp(cv_scores)  # 极差
                
                # 显示结果
                st.subheader("交叉验证结果")
                col1, col2, col3 = st.columns(3)
                col1.metric("平均 R²", f"{mean_score:.4f}")
                col2.metric("标准差", f"{std_score:.4f}")
                col3.metric("极差", f"{cv_range:.4f}")
                
                # 绘制结果分布
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(cv_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                ax.axvline(mean_score, color='red', linestyle='--', label=f'平均值: {mean_score:.4f}')
                ax.set_xlabel('R方 分数')
                ax.set_ylabel('频数')
                ax.set_title(f'交叉验证 R方 分数分布 ({n_folds}折 × {n_repeats}次)')
                ax.legend()
                st.pyplot(fig)
                
                # 稳定性评估
                st.subheader("稳定性评估")
                if std_score < 0.05:
                    stability = "高稳定性"
                    stability_color = "green"
                    stability_desc = "模型在不同数据子集上表现一致，稳定性良好"
                elif std_score < 0.1:
                    stability = "中等稳定性"
                    stability_color = "orange"
                    stability_desc = "模型表现有一定波动，但总体稳定"
                else:
                    stability = "低稳定性"
                    stability_color = "red"
                    stability_desc = "模型表现波动较大，可能对数据敏感或过拟合"
                
                st.markdown(f"<h4 style='color:{stability_color}'>稳定性: {stability}</h4>", unsafe_allow_html=True)
                st.write(stability_desc)
                
                # 提供改进建议
                st.subheader("改进建议")
                if stability == "低稳定性":
                    st.write("""
                    - 尝试增加数据量或使用数据增强
                    - 简化模型复杂度（如减少树的最大深度）
                    - 增加正则化强度
                    - 检查特征选择是否合理
                    - 考虑使用更稳定的算法（如线性模型）
                    """)
                elif stability == "中等稳定性":
                    st.write("""
                    - 可以尝试调整模型超参数
                    - 考虑使用集成方法提高稳定性
                    - 检查是否有异常值影响模型
                    """)
                else:
                    st.write("""
                    - 模型稳定性良好，可以放心使用
                    - 可以考虑进一步优化模型性能
                    """)
    
    elif validation_method == "自助法":
        st.subheader("自助法稳定性分析")
        
        n_iterations = st.slider("自助法迭代次数", 10, 100, 50)
        sample_ratio = st.slider("每次采样的比例", 0.5, 0.95, 0.8)
        
        if st.button("开始自助法验证"):
            with st.spinner("正在进行自助法验证..."):
                # 获取数据和模型
                df = st.session_state.data
                features = st.session_state.features
                target = st.session_state.target
                
                X = df[features]
                y = df[target]
                
                # 存储每次迭代的得分
                bootstrap_scores = []
                
                # 进度条
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(n_iterations):
                    # 更新进度
                    progress = (i + 1) / n_iterations
                    progress_bar.progress(progress)
                    status_text.text(f"正在进行第 {i+1}/{n_iterations} 次迭代...")
                    
                    # 自助法采样
                    n_samples = int(len(X) * sample_ratio)
                    indices = np.random.choice(len(X), n_samples, replace=True)
                    X_bootstrap = X.iloc[indices]
                    y_bootstrap = y.iloc[indices]
                    
                    # 划分训练测试集
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_bootstrap, y_bootstrap, test_size=0.2, random_state=42
                    )
                    
                    # 克隆原始模型
                    from sklearn.base import clone
                    model_clone = clone(st.session_state.model)
                    
                    # 训练和评估
                    model_clone.fit(X_train, y_train)
                    y_pred = model_clone.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    bootstrap_scores.append(score)
                
                # 完成进度
                progress_bar.empty()
                status_text.empty()
                
                # 计算统计量
                mean_score = np.mean(bootstrap_scores)
                std_score = np.std(bootstrap_scores)
                ci_low = np.percentile(bootstrap_scores, 2.5)
                ci_high = np.percentile(bootstrap_scores, 97.5)
                
                # 显示结果
                st.subheader("自助法验证结果")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("平均 R²", f"{mean_score:.4f}")
                col2.metric("标准差", f"{std_score:.4f}")
                col3.metric("95% CI 下限", f"{ci_low:.4f}")
                col4.metric("95% CI 上限", f"{ci_high:.4f}")
                
                # 绘制结果分布
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(bootstrap_scores, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax.axvline(mean_score, color='red', linestyle='--', label=f'平均值: {mean_score:.4f}')
                ax.axvline(ci_low, color='orange', linestyle=':', label=f'95% CI下限: {ci_low:.4f}')
                ax.axvline(ci_high, color='orange', linestyle=':', label=f'95% CI上限: {ci_high:.4f}')
                ax.set_xlabel('R方 分数')
                ax.set_ylabel('频数')
                ax.set_title(f'自助法 R方 分数分布 ({n_iterations}次迭代)')
                ax.legend()
                st.pyplot(fig)
                
                # 稳定性评估
                st.subheader("稳定性评估")
                score_range = ci_high - ci_low
                if score_range < 0.1:
                    stability = "高稳定性"
                    stability_color = "green"
                    stability_desc = "模型在不同数据子集上表现一致，置信区间窄"
                elif score_range < 0.2:
                    stability = "中等稳定性"
                    stability_color = "orange"
                    stability_desc = "模型表现有一定波动，但置信区间合理"
                else:
                    stability = "低稳定性"
                    stability_color = "red"
                    stability_desc = "模型表现波动较大，置信区间宽"
                
                st.markdown(f"<h4 style='color:{stability_color}'>稳定性: {stability}</h4>", unsafe_allow_html=True)
                st.write(stability_desc)
                st.write(f"95% 置信区间宽度: {score_range:.4f}")
                st.sidebar.markdown("---")
st.sidebar.info(
    """
    **使用说明:** 
    1. 在「实验数据插值增强」页面进行基于物理直觉的数据增强（可选）
    2. 在「数据上传与设置」页面上传CSV数据并设置特征
    3. 在「模型训练」页面训练"随机森林回归", "线性回归", "Ridge回归", "Lasso回归", "支持向量回归(SVR)"模型
    4. 在「参数预测」页面输入参数进行预测
    5. 在「特征分析」页面查看各参数的重要性以及参数间关系
    6. 在「参数分布分析」页面查看参数对观测量的分布影响
    7. 在「模型稳定性验证」页面评估模型的稳定性
    8. 所有结果均可下载，方便后续分析
    """
)