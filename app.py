# app.py
import os
import numpy as np
import pandas as pd
import streamlit as st

# sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve, silhouette_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# vis
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Doença Cardíaca — Relatório + EDA + Modelos", layout="wide")

# ============================================
# CARREGAR DADOS (com fallback de caminho)
# ============================================
@st.cache_data
def carregar_dados():
    if os.path.exists("heart.csv"):
        return pd.read_csv("heart.csv")
    elif os.path.exists("/mnt/data/heart.csv"):
        return pd.read_csv("/mnt/data/heart.csv")
    else:
        raise FileNotFoundError("Arquivo 'heart.csv' não encontrado.")

df = carregar_dados().copy()
df["target"] = df["target"].astype(int)

# --------------------------------------------
# Dicionários de rótulos (display) e inversos
# --------------------------------------------
sex_map     = {1: "Masculino", 0: "Feminino"}
fbs_map     = {1: "Glicose > 120 mg/dl", 0: "Normal"}
exang_map   = {1: "Sim", 0: "Não"}
restecg_map = {0: "Normal", 1: "Anomalia de ST-T", 2: "Hipertrofia Ventricular"}
slope_map   = {0: "Descendente", 1: "Plano", 2: "Ascendente"}
cp_map      = {0: "Angina típica", 1: "Angina atípica", 2: "Dor não anginosa", 3: "Assintomático"}
thal_map    = {0: "Desconhecido", 1: "Normal", 2: "Defeito fixo", 3: "Defeito reversível"}

# inversos
sex_inv     = {v:k for k,v in sex_map.items()}
fbs_inv     = {v:k for k,v in fbs_map.items()}
exang_inv   = {v:k for k,v in exang_map.items()}
restecg_inv = {v:k for k,v in restecg_map.items()}
slope_inv   = {v:k for k,v in slope_map.items()}
cp_inv      = {v:k for k,v in cp_map.items()}
thal_inv    = {v:k for k,v in thal_map.items()}

# DataFrame só para visualização com rótulos legíveis
df_disp = df.copy()
if "sex" in df_disp:     df_disp["sex"]     = df_disp["sex"].map(sex_map)
if "fbs" in df_disp:     df_disp["fbs"]     = df_disp["fbs"].map(fbs_map)
if "exang" in df_disp:   df_disp["exang"]   = df_disp["exang"].map(exang_map)
if "restecg" in df_disp: df_disp["restecg"] = df_disp["restecg"].map(restecg_map)
if "slope" in df_disp:   df_disp["slope"]   = df_disp["slope"].map(slope_map)
if "cp" in df_disp:      df_disp["cp"]      = df_disp["cp"].map(cp_map)
if "thal" in df_disp:    df_disp["thal"]    = df_disp["thal"].map(thal_map)
# 'ca' permanece numérico (0–4) → número de vasos principais coloridos por fluoroscopia

# Colunas
NUM_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CAT_COLS = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
ALL_FEATURES = NUM_COLS + CAT_COLS
X = df[ALL_FEATURES]
y = df['target']

# Pré-processador comum (numéricas padronizadas + categóricas one-hot)
preprocess_common = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ],
    remainder="drop"
)

# bounds p/ validação leve (p5–p95) das numéricas
bounds = {c: (float(np.nanpercentile(df[c], 5)), float(np.nanpercentile(df[c], 95))) for c in NUM_COLS}
def clamp_num(name, val):
    lo, hi = bounds.get(name, (None, None))
    return float(np.clip(val, lo, hi)) if lo is not None else float(val)

# ============================================
# ABAS — nova ordem:
# 1) Relatório, 2) EDA, 3) Superv., 4) Não Superv.
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📑 Relatório Automático de Insights",
    "📊 Exploratória (turbinada)",
    "💖 Supervisionado (RandomForest)",
    "🧠 Não Supervisionado (KMeans)"
])

# ============================================
# TAB 1 — RELATÓRIO AUTOMÁTICO
# ============================================
with tab1:
    st.title("📑 Relatório Automático de Insights")

    # 1) Balanceamento
    pos_rate = df["target"].mean()*100
    st.subheader("🎯 Balanceamento do Alvo")
    st.write(f"Proporção de pacientes com doença (target=1): **{pos_rate:.1f}%**.")

    # 2) Top correlações com o alvo (numéricas)
    st.subheader("🔗 Top correlações (numéricas) com o alvo")
    corr_target = df[NUM_COLS + ["target"]].corr()["target"].drop("target").sort_values(ascending=False)
    st.write(corr_target.to_frame("correlação").round(3))

    # 3) Diferenças por categoria (taxa de target=1) — usando df original (numérico)
    st.subheader("🏷️ Variáveis categóricas que mais diferenciam o alvo")
    diffs = []
    for c in CAT_COLS:
        rates = df.groupby(c)["target"].mean()
        diffs.append((c, float(rates.max() - rates.min())))
    diffs = sorted(diffs, key=lambda x: x[1], reverse=True)
    st.write(pd.DataFrame(diffs, columns=["variável", "Δ máxima de proporção"]).head(10))

    # 4) Possíveis outliers numéricos (z-score simples)
    st.subheader("🚩 Possíveis outliers (numéricos)")
    outs = {}
    for c in NUM_COLS:
        z = (df[c] - df[c].mean())/df[c].std(ddof=0)
        outs[c] = int((np.abs(z) > 3).sum())
    st.write(pd.DataFrame.from_dict(outs, orient="index", columns=["contagem_outliers"]))

    # 5) Recomendação de modelagem
    st.subheader("🧠 Recomendações de Modelagem")
    st.markdown("""
    - **Supervisionado**: Utilize **RandomForest** (aba dedicada); para interpretabilidade adicional, teste **Regressão Logística** com One-Hot.
    - **Não supervisionado**: **KMeans** com **k sugerido por Silhouette** (aba dedicada); para clusters elípticos, avalie **GaussianMixture**.
    - **Validação**: *Hold-out* estratificado e monitoramento de **AUC/ROC**, **Precision/Recall** conforme a necessidade.
    - **Limiar**: Ajuste por **Youden (ROC)** no supervisionado; no não-supervisionado use **baseline** ou **mediana** dos riscos dos clusters.
    """)

    # 6) Próximos passos
    st.subheader("🔭 Próximos passos sugeridos")
    st.markdown("""
    - Curva ROC interativa para ajuste de limiar.
    - Explicabilidade com SHAP/PDP.
    - Busca de hiperparâmetros (RandomizedSearch/Optuna).
    - Comparar KMeans vs GMM (Silhouette/BIC).
    """)

# ============================================
# TAB 2 — EXPLORATÓRIA (aprimorada) — usa df_disp (rótulos)
# ============================================
with tab2:
    st.title("📊 Análise Exploratória dos Dados (EDA)")

    # Overview
    st.header("📋 Visão Geral")
    col1, col2, col3 = st.columns(3)
    col1.metric("Observações", df.shape[0])
    col2.metric("Variáveis (total)", df.shape[1])
    col3.metric("Proporção com doença (target=1)", f"{df['target'].mean()*100:.1f}%")

    st.subheader("📈 Estatísticas Descritivas (Numéricas)")
    st.dataframe(df[NUM_COLS + ["target"]].describe().T.style.background_gradient(cmap="Blues"))

    # Distribuições (com rótulos para categóricas)
    st.header("📊 Distribuições e Comparações por Diagnóstico")
    ALL_FOR_DISPLAY = NUM_COLS + CAT_COLS
    var = st.selectbox("Escolha uma variável:", ALL_FOR_DISPLAY, index=0, key="eda_var")

    if var in NUM_COLS:
        fig = px.histogram(df, x=var, color="target", nbins=30, barmode="overlay",
                           title=f"Distribuição de {var} por Diagnóstico (target)")
        st.plotly_chart(fig, use_container_width=True)

        colA, colB = st.columns(2)
        with colA:
            fig = px.box(df, x="target", y=var, color="target", points="all",
                         title=f"Boxplot de {var} por target")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            fig = px.violin(df, x="target", y=var, color="target", box=True, points="all",
                            title=f"Violin plot de {var} por target")
            st.plotly_chart(fig, use_container_width=True)
    else:
        # usar df_disp para rótulos legíveis
        fig = px.histogram(df_disp, x=var, color="target", barmode="group",
                           title=f"Contagem de {var} por Diagnóstico (target) — rótulos legíveis")
        st.plotly_chart(fig, use_container_width=True)

    # Correlações (numéricas)
    st.header("🔗 Correlação entre Variáveis (numéricas)")
    corr = df[NUM_COLS + ["target"]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)
    st.caption("Correlação de Pearson entre variáveis numéricas e o diagnóstico (`target`).")

    # Relações numéricas com tendência (sem statsmodels)
    st.header("📈 Relações Numéricas com Tendência")
    colx, coly = st.columns(2)
    with colx:
        varx = st.selectbox("Eixo X", NUM_COLS, index=0)
    with coly:
        vary = st.selectbox("Eixo Y", NUM_COLS, index=1)
    fig_sc = px.scatter(df, x=varx, y=vary, color="target",
                        title=f"Relação entre {varx} e {vary} (com linha de tendência)")
    xvals = df[varx].values
    yvals = df[vary].values
    if np.isfinite(xvals).all() and np.isfinite(yvals).all():
        slope, intercept = np.polyfit(xvals, yvals, 1)
        xs = np.linspace(xvals.min(), xvals.max(), 100)
        ys = slope*xs + intercept
        fig_sc.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Tendência (global)"))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Comparações categóricas com rótulos legíveis
    st.header("🏷️ Comparações Categóricas")
    cat_var = st.selectbox("Variável categórica:", CAT_COLS, index=0, key="eda_cat")
    resumo = df.groupby(cat_var)["target"].agg(["mean", "count"]).rename(
        columns={"mean": "Proporção de doença", "count": "N"}
    )
    # legenda amigável
    legenda = {
        "sex": sex_map, "fbs": fbs_map, "exang": exang_map, "restecg": restecg_map,
        "slope": slope_map, "cp": cp_map, "thal": thal_map
    }
    if cat_var in legenda:
        resumo = resumo.copy()
        resumo.index = resumo.index.map(legenda[cat_var])
    st.write(resumo)
    fig = px.bar(resumo, x=resumo.index, y="Proporção de doença", color=resumo.index,
                 title=f"Taxa média de doença por {cat_var} (rótulos legíveis)")
    st.plotly_chart(fig, use_container_width=True)

    # Interação multicategórica (rótulos sex/cp)
    st.header("🔍 Interação: Sexo × Tipo de Dor (cp) × Diagnóstico")
    pivot = df.pivot_table(values="target", index="cp", columns="sex", aggfunc="mean")
    pivot.index = pivot.index.map(cp_map)
    pivot.columns = pivot.columns.map(sex_map)
    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="Reds",
                    labels=dict(x="Sexo", y="Tipo de Dor (cp)", color="Prob. de Doença"))
    st.plotly_chart(fig, use_container_width=True)

    # PCA 3D (numéricas)
    st.header("🧩 PCA (Redução de Dimensionalidade, numéricas)")
    pca = PCA(n_components=3, random_state=42)
    X_num_scaled = StandardScaler().fit_transform(df[NUM_COLS])
    X_pca = pca.fit_transform(X_num_scaled)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
    df_pca["target"] = df["target"]
    st.caption(f"Variância explicada: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
               f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%, "
               f"PC3={pca.explained_variance_ratio_[2]*100:.1f}%")
    fig = px.scatter_3d(df_pca, x="PC1", y="PC2", z="PC3", color="target",
                        title="Projeção PCA 3D — separação por diagnóstico")
    st.plotly_chart(fig, use_container_width=True)

# ============================================
# TAB 3 — SUPERVISIONADO (RandomForest) — selects com rótulos
# ============================================
with tab3:
    st.title("💖 Predição (RandomForest + OneHot + StandardScaler)")

    # Controles
    colM1, colM2, colM3 = st.columns(3)
    with colM1:
        n_estimators = st.slider("n_estimators", 100, 1000, 400, step=50)
    with colM2:
        max_depth = st.slider("max_depth (None=0)", 0, 30, 0)
        max_depth = None if max_depth == 0 else max_depth
    with colM3:
        test_size = st.slider("Tamanho do teste (%)", 10, 40, 25, step=5) / 100.0

    # Split + pipeline
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1
    )
    pipe_rf = Pipeline(steps=[("prep", preprocess_common), ("rf", rf)])
    pipe_rf.fit(X_train, y_train)

    # Limiar
    st.markdown("**Limiar de classificação**")
    limiar_mode = st.selectbox("Modo do limiar", ["0.5 (padrão)", "Automático (Youden/ROC)"])
    y_proba_test = pipe_rf.predict_proba(X_test)[:,1]
    if limiar_mode == "Automático (Youden/ROC)":
        fpr, tpr, thr_grid = roc_curve(y_test, y_proba_test)
        j = tpr - fpr
        thr = float(thr_grid[np.argmax(j)])
    else:
        thr = 0.5
    st.info(f"Limiar atual: **{thr:.3f}**")

    # Formulário com rótulos legíveis
    with st.form("form_superv"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Idade (age)", 0, 120, int(df["age"].median()))
            trestbps = st.number_input("Pressão em repouso (trestbps)", 50, 260, int(df["trestbps"].median()))
            chol = st.number_input("Colesterol (chol)", 80, 600, int(df["chol"].median()))
        with c2:
            thalach = st.number_input("Frequência máx. (thalach)", 40, 250, int(df["thalach"].median()))
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, float(df["oldpeak"].median()), step=0.1)
            sex_label = st.selectbox("Sexo (sex)", list(sex_inv.keys()), index=0 if df['sex'].mode().iat[0]==1 else 1)
        with c3:
            cp_label = st.selectbox("Tipo de dor no peito (cp)", list(cp_inv.keys()))
            fbs_label = st.selectbox("Glicose em jejum (fbs)", list(fbs_inv.keys()))
            restecg_label = st.selectbox("ECG de repouso (restecg)", list(restecg_inv.keys()))
            exang_label = st.selectbox("Angina induzida por exercício (exang)", list(exang_inv.keys()))
            slope_label = st.selectbox("Inclinação ST no pico (slope)", list(slope_inv.keys()))
            ca = st.selectbox("Vasos principais coloridos (ca)", [0,1,2,3,4], help="Número de vasos (0–4)")
            thal_label = st.selectbox("Thalassemia (thal)", list(thal_inv.keys()))

        pred_click = st.form_submit_button("🔮 Prever (Supervisionado)")

    if pred_click:
        # mapear de volta para os códigos numéricos
        novo = pd.DataFrame([{
            "age": clamp_num("age", age),
            "trestbps": clamp_num("trestbps", trestbps),
            "chol": clamp_num("chol", chol),
            "thalach": clamp_num("thalach", thalach),
            "oldpeak": clamp_num("oldpeak", oldpeak),
            "sex": sex_inv[sex_label],
            "cp": cp_inv[cp_label],
            "fbs": fbs_inv[fbs_label],
            "restecg": restecg_inv[restecg_label],
            "exang": exang_inv[exang_label],
            "slope": slope_inv[slope_label],
            "ca": ca,
            "thal": thal_inv[thal_label]
        }])
        prob = float(pipe_rf.predict_proba(novo)[0,1])
        pred = int(prob >= thr)

        st.markdown("---")
        st.subheader("🩺 Resultado (Supervisionado)")
        if pred == 1:
            st.error(f"🚨 **Alto risco** de doença cardíaca ({prob*100:.1f}%)")
        else:
            st.success(f"💚 **Baixo risco** de doença cardíaca ({prob*100:.1f}%)")
        st.progress(prob)

        out = novo.copy()
        out["prob"] = prob
        out["pred"] = pred
        st.download_button(
            "⬇️ Baixar resultado (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="pred_supervisionado.csv",
            mime="text/csv"
        )

    # Métricas no teste
    y_pred_test = (y_proba_test >= thr).astype(int)
    st.markdown("---")
    st.subheader("📊 Desempenho")
    cA, cB, cC = st.columns(3)
    with cA: st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_test):.2%}")
    with cB: st.metric("ROC AUC", f"{roc_auc_score(y_test, y_proba_test):.3f}")
    with cC:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        st.metric("TP / FP / FN / TN", f"{tp} / {fp} / {fn} / {tn}")

    fig_cm, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt="d", cmap="Reds",
                xticklabels=["Sem", "Com"], yticklabels=["Sem", "Com"], ax=ax)
    ax.set_title("Matriz de Confusão")
    st.pyplot(fig_cm)

# ============================================
# TAB 4 — NÃO SUPERVISIONADO (KMeans) — selects com rótulos
# ============================================
with tab4:
    st.title("🧠 Agrupamento (KMeans + OneHot + StandardScaler)")

    # Dados transformados p/ clustering
    X_transformed = preprocess_common.fit_transform(X)

    # Sugerir k por silhouette (2..8)
    with st.spinner("Avaliando k por Silhouette (2–8)..."):
        silscores = {}
        for kk in range(2, 9):
            km = KMeans(n_clusters=kk, random_state=42, n_init=10)
            labels = km.fit_predict(X_transformed)
            silscores[kk] = silhouette_score(X_transformed, labels)
        k_sugerido = max(silscores, key=silscores.get)

    colK1, colK2, colK3 = st.columns(3)
    with colK1: st.metric("k sugerido", k_sugerido, help="Maximiza o Silhouette Score")
    with colK2: st.metric("Silhouette(k sugerido)", f"{silscores[k_sugerido]:.3f}")
    with colK3: k = st.slider("Escolha k (pode sobrepor o sugerido)", 2, 10, int(k_sugerido))

    # Treina KMeans final
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_transformed)
    df_clusters = df.copy()
    df_clusters["Cluster"] = clusters

    # Métricas globais
    baseline = float(df["target"].mean() * 100)
    sil_final = float(silhouette_score(X_transformed, clusters))
    st.caption(f"Baseline (positivos no dataset): {baseline:.1f}% • Silhouette(k={k}): {sil_final:.3f}")

    # Risco por cluster
    risco_por_cluster = df_clusters.groupby("Cluster")["target"].mean().reindex(range(k), fill_value=np.nan) * 100
    st.write("**Risco médio por cluster (%)**")
    st.dataframe(risco_por_cluster.round(1).to_frame("risco_%").T)

    # PCA 2D para visualização
    st.subheader("PCA (2D) dos clusters")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_transformed)
    vis = pd.DataFrame({"PCA1": X_pca[:,0], "PCA2": X_pca[:,1], "Cluster": df_clusters["Cluster"].astype(str)})
    fig_pca = px.scatter(vis, x="PCA1", y="PCA2", color="Cluster", title="Distribuição por Cluster (PCA)")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Formulário com rótulos legíveis
    with st.form("form_unsup"):
        u1, u2, u3 = st.columns(3)
        with u1:
            age = st.number_input("Idade (age)", 0, 120, int(df["age"].median()), key="u_age")
            trestbps = st.number_input("Pressão em repouso (trestbps)", 50, 260, int(df["trestbps"].median()), key="u_trestbps")
            chol = st.number_input("Colesterol (chol)", 80, 600, int(df["chol"].median()), key="u_chol")
        with u2:
            thalach = st.number_input("Frequência máx. (thalach)", 40, 250, int(df["thalach"].median()), key="u_thalach")
            oldpeak = st.number_input("ST depression (oldpeak)", 0.0, 10.0, float(df["oldpeak"].median()), step=0.1, key="u_oldpeak")
            sex_label = st.selectbox("Sexo (sex)", list(sex_inv.keys()), index=0 if df['sex'].mode().iat[0]==1 else 1, key="u_sex")
        with u3:
            cp_label = st.selectbox("Tipo de dor (cp)", list(cp_inv.keys()), key="u_cp")
            fbs_label = st.selectbox("Glicose em jejum (fbs)", list(fbs_inv.keys()), key="u_fbs")
            restecg_label = st.selectbox("ECG repouso (restecg)", list(restecg_inv.keys()), key="u_restecg")
            exang_label = st.selectbox("Angina por exercício (exang)", list(exang_inv.keys()), key="u_exang")
            slope_label = st.selectbox("Inclinação ST (slope)", list(slope_inv.keys()), key="u_slope")
            ca = st.selectbox("Vasos principais coloridos (ca)", [0,1,2,3,4], help="Número de vasos (0–4)", key="u_ca")
            thal_label = st.selectbox("Thalassemia (thal)", list(thal_inv.keys()), key="u_thal")

        limiar_mode = st.selectbox("Limiar Alto Risco",
                                   ["Manual (50%)", "Baseline do dataset", "Mediana dos riscos dos clusters"])
        limiar_manual = st.slider("Limiar manual (%)", 10, 90, 50)

        calc_click = st.form_submit_button("🧠 Calcular Cluster & Risco")

    if calc_click:
        # mapear de volta para os códigos numéricos
        novo = pd.DataFrame([{
            "age": clamp_num("age", age),
            "trestbps": clamp_num("trestbps", trestbps),
            "chol": clamp_num("chol", chol),
            "thalach": clamp_num("thalach", thalach),
            "oldpeak": clamp_num("oldpeak", oldpeak),
            "sex": sex_inv[sex_label],
            "cp": cp_inv[cp_label],
            "fbs": fbs_inv[fbs_label],
            "restecg": restecg_inv[restecg_label],
            "exang": exang_inv[exang_label],
            "slope": slope_inv[slope_label],
            "ca": ca,
            "thal": thal_inv[thal_label]
        }])
        novo_transf = preprocess_common.transform(novo)
        cluster_pred = int(kmeans.predict(novo_transf)[0])

        risco_cluster = float(df_clusters.loc[df_clusters["Cluster"]==cluster_pred, "target"].mean() * 100)
        cluster_size = int((df_clusters["Cluster"]==cluster_pred).sum())
        delta = risco_cluster - baseline

        # Limiar
        if limiar_mode == "Baseline do dataset":
            limiar = baseline
        elif limiar_mode == "Mediana dos riscos dos clusters":
            limiar = float(np.nanmedian(risco_por_cluster.values))
        else:
            limiar = float(limiar_manual)

        st.markdown("---")
        st.subheader("🧩 Resultado (Não Supervisionado)")
        msg = f"Cluster {cluster_pred} — risco médio ({risco_cluster:.1f}%)"
        if risco_cluster >= limiar:
            st.error(f"🚨 **Alto risco** • {msg}")
        else:
            st.success(f"💚 **Baixo risco** • {msg}")
        st.progress(risco_cluster/100.0)

        c1, c2, c3 = st.columns(3)
        c1.metric("Risco do Cluster", f"{risco_cluster:.1f}%", delta=f"{delta:+.1f} pts vs. baseline")
        c2.metric("Baseline do Dataset", f"{baseline:.1f}%")
        c3.metric("Tamanho do Cluster", f"{cluster_size}")

        out = novo.copy()
        out["cluster_pred"] = cluster_pred
        out["risco_cluster_pct"] = risco_cluster
        out["limiar_modo"] = limiar_mode
        out["limiar_pct"] = limiar
        st.download_button(
            "⬇️ Baixar resultado (CSV)",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="cluster_resultado.csv",
            mime="text/csv"
        )

# Rodapé
st.caption("⚠️ Uso educacional/analítico — não substitui avaliação clínica.")
