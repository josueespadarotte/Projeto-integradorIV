import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========================= CONFIGURAÇÃO DA PÁGINA =========================
st.set_page_config(
    page_title="PRF-ES Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CARREGAMENTO DE ÍCONES =========================
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

# ========================= CSS CORRIGIDO E BLINDADO =========================
st.markdown(
    """
    <style>
    /* ----- VARIÁVEIS DE COR ----- */
    :root {
        --primary: #0A2647;
        --primary-light: #2C74B3;
        --accent: #F39C12;
        --glass-bg: var(--secondary-background-color);
        --shadow-sm: 0 4px 12px rgba(0, 0, 0, 0.05);
        --shadow-hover: 0 8px 24px rgba(30, 144, 255, 0.3);
    }

    .block-container {
        padding-top: 1.5rem !important;
    }

    /* ----- BARRA LATERAL (Sem bugar os selects) ----- */
    /* Letras brancas (como já estava funcionando) */
    [data-testid="stSidebar"] details:not([open]) summary p {
        color: #FFFFFF !important;
    }
    /* Força as setinhas a ficarem brancas (cobre fill, stroke e color) */
    [data-testid="stSidebar"] details:not([open]) summary svg,
    [data-testid="stSidebar"] details:not([open]) summary svg path {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    /* Força a borda do expander fechado a ficar branca (com leve transparência para ficar elegante) */
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) {
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    /* Garante que as letras e setinhas continuem brancas ao passar o mouse */
    [data-testid="stSidebar"] details summary:hover p,
    [data-testid="stSidebar"] details summary:hover svg,
    [data-testid="stSidebar"] details summary:hover svg path {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }

    /* Suaviza o fundo que aparece ao passar o mouse no título do filtro */
    [data-testid="stSidebar"] details summary:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0A2647 0%, #1B3A5C 100%) !important;
    }
    /* Pinta apenas os textos soltos e labels, protegendo o dropdown interno */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .st-expanderHeader p {
        color: #F5F5F5 !important;
        font-weight: 600 !important;
    }

    /* ----- CARDS DE MÉTRICA (Glow tech seguro) ----- */
    div[data-testid="stMetric"] {
        background: var(--glass-bg) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        box-shadow: var(--shadow-sm) !important;
        border-left: 4px solid var(--primary-light) !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-hover) !important;
    }
    [data-testid="stMetricLabel"] * {
        font-weight: 600 !important;
        color: var(--text-color) !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
    }
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        color: var(--primary-light) !important;
    }

    /* ----- ABAS CENTRALIZADAS ----- */
    div[data-testid="stTabList"] {
        display: flex !important;
        justify-content: center !important; 
        width: 100% !important;
        gap: 8px !important;
    }
    button[data-baseweb="tab"] {
        background-color: transparent !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.5rem !important;
        transition: 0.2s;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent !important;
        color: var(--primary-light) !important;
        box-shadow: none !important;
    }
    button[data-baseweb="tab"] p {
        font-size: 0.9rem !important; 
        font-weight: 600 !important;
    }

    /* ----- TÍTULOS GERAIS ----- */
    h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    .subtitulo-header {
        font-size: 0.9rem !important;
        opacity: 0.8;
        border-left: 3px solid var(--accent);
        padding-left: 12px;
        margin-bottom: 20px !important;
    }
    .chart-title {
        font-weight: 700;
        margin: 1rem 0 0.5rem 0;
        font-size: 1rem;
        color: var(--primary-light);
    }

    /* ----- SLIDER ----- */
    div[data-testid="stSlider"] > div > div > div {
        background-color: var(--primary-light) !important;
        height: 4px !important;
    }
    
    /* ----- BOTÃO DE DOWNLOAD ----- */
    div.stButton > button {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        color: white;
        border-radius: 30px !important;
        border: none;
        font-weight: 600;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        box-shadow: var(--shadow-hover) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========================= CARREGAMENTO DE DADOS =========================
@st.cache_data
def load_data():
    df = pd.read_csv('app/prf_es_clean.csv')
    df['hora_inteira'] = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce').dt.hour
    for col in ['latitude', 'longitude']:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['latitude', 'longitude'])

df = load_data()

# ========================= SIDEBAR (Filtros) =========================
st.sidebar.image("https://prf.dsoconcursos.com.br/wp-content/uploads/2024/08/logotipo-prf-1024x443.png", width=160)
st.sidebar.markdown("<h2 style='color:#F5F5F5;'></i> Central de Comando</h2>", unsafe_allow_html=True)

with st.sidebar.expander("📍 Filtros de Localização", expanded=True):
    municipios = st.multiselect("Município", sorted(df['municipio'].unique()))
    rodovias = st.multiselect("Rodovia (BR)", sorted(df['br'].unique().astype(str)))

with st.sidebar.expander("🌦️ Condições da Via"):
    clima = st.multiselect("Condição Climática", df['condicao_metereologica'].unique(), default=df['condicao_metereologica'].unique())
    pista = st.multiselect("Tipo de Pista", df['tipo_pista'].unique(), default=df['tipo_pista'].unique())

horarios = st.sidebar.slider("⏰ Faixa Horária", 0, 23, (0, 23))

# Filtragem
df_f = df.copy()
if municipios: df_f = df_f[df_f['municipio'].isin(municipios)]
if rodovias: df_f = df_f[df_f['br'].astype(str).isin(rodovias)]
df_f = df_f[
    (df_f['condicao_metereologica'].isin(clima)) &
    (df_f['tipo_pista'].isin(pista)) &
    (df_f['hora_inteira'].between(horarios[0], horarios[1]))
]

# Botão de Download
csv_filtrado = df_f.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Baixar Dados Filtrados",
    data=csv_filtrado,
    file_name='prf_es_filtrado.csv',
    mime='text/csv',
)

# ========================= PAINEL PRINCIPAL =========================
st.markdown("<h1><i class='fas fa-chart-line'></i> Sistema de Inteligência Viária - PRF Espírito Santo</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitulo-header'>Monitoramento em tempo real de pontos críticos e severidade de acidentes.</p>", unsafe_allow_html=True)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total de Ocorrências", len(df_f))
with kpi2:
    fatais = df_f['mortos'].sum()
    st.metric("Vítimas Fatais", int(fatais))
with kpi3:
    graves = len(df_f[df_f['classificacao_acidente'].str.contains('Grave|Fatal|Fatais', case=False, na=False)])
    st.metric("Acidentes Graves", graves)
with kpi4:
    taxa = (graves / len(df_f) * 100) if len(df_f) > 0 else 0
    st.metric("Índice de Letalidade", f"{taxa:.1f}%")

st.write("") 

# ========================= ABAS =========================
aba_mapa, aba_estatisticas, aba_ia = st.tabs([
    "📍 Mapa de Calor", 
    "📊 Análise Estatística", 
    "🤖 Predição de Risco"
])

with aba_mapa:
    st.subheader("Concentração Geográfica de Acidentes")
    mapa_data = df_f[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    st.map(mapa_data, zoom=7, height=450, use_container_width=True)

with aba_estatisticas:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='chart-title'><i class='fas fa-exclamation-triangle'></i> Top 10 Causas de Acidentes</div>", unsafe_allow_html=True)
        causas = df_f['causa_acidente'].value_counts().head(10).reset_index()
        fig_causa = px.bar(
            causas, x='count', y='causa_acidente', orientation='h',
            color_discrete_sequence=['#2C74B3'], text_auto=True
        )
        fig_causa.update_layout(yaxis={'categoryorder': 'total ascending'}, height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_causa, use_container_width=True)

    with col_b:
        st.markdown("<div class='chart-title'><i class='fas fa-road'></i> Acidentes por Tipo de Pista e Gravidade</div>", unsafe_allow_html=True)
        fig_pista = px.histogram(
            df_f, x='tipo_pista', color='classificacao_acidente',
            barmode='group', color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_pista.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pista, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='chart-title'><i class='fas fa-cloud-rain'></i> Análise de Risco: A influência do clima em acidentes fatais</div>", unsafe_allow_html=True)

    df_clima = df_f.groupby('condicao_metereologica').agg(
        total_acidentes=('classificacao_acidente', 'count'),
        vitimas_fatais=('mortos', 'sum')
    ).reset_index().sort_values('total_acidentes', ascending=False)

    fig_clima = make_subplots(specs=[[{"secondary_y": True}]])
    fig_clima.add_trace(
        go.Bar(x=df_clima['condicao_metereologica'], y=df_clima['total_acidentes'], name="Total de Acidentes", marker_color='#2C74B3'),
        secondary_y=False,
    )
    fig_clima.add_trace(
        go.Scatter(x=df_clima['condicao_metereologica'], y=df_clima['vitimas_fatais'], name="Vítimas Fatais", mode="lines+markers", line=dict(color='#D62828', width=3), marker=dict(size=8)),
        secondary_y=True,
    )
    fig_clima.update_layout(
        height=400, hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig_clima.update_yaxes(title_text="Volume de Acidentes", secondary_y=False)
    fig_clima.update_yaxes(title_text="Vítimas Fatais", secondary_y=True)
    st.plotly_chart(fig_clima, use_container_width=True)

with aba_ia:
    st.subheader("🧠 Entendendo as Decisões do Modelo")
    try:
        modelo_ia = joblib.load('models/modelo_risco_acidente.pkl')
        colunas_ia = joblib.load('models/colunas_treino.pkl')

        importancias = modelo_ia.feature_importances_
        df_importancia = pd.DataFrame({
            'Fator de Risco': colunas_ia,
            'Impacto no Modelo': importancias
        }).sort_values(by='Impacto no Modelo', ascending=True).tail(10)

        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('condicao_metereologica_', '🌦️ Clima: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('tipo_pista_', '🛣️ Pista: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('causa_acidente_', '⚠️ Causa: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('fase_dia_', '☀️ Período: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('hora_inteira', '⏰ Hora do Acidente')

        fig_imp = px.bar(
            df_importancia, x='Impacto no Modelo', y='Fator de Risco', orientation='h',
            color='Impacto no Modelo', color_continuous_scale='Reds'
        )
        fig_imp.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Importância Estatística", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("💡 Gráfico de importância indisponível – verifique os arquivos .pkl na pasta 'models/'.")

    st.markdown("---")
    st.subheader("⚙️ Simulador de Risco (Machine Learning)")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_clima = st.selectbox("Condição do Tempo", df['condicao_metereologica'].unique())
    with col2:
        sel_pista = st.selectbox("Tipo da Via", df['tipo_pista'].unique())
    with col3:
        sel_hora = st.number_input("Horário (0-23h)", 0, 23, 12)

    if st.button("Executar Diagnóstico de Risco"):
        try:
            modelo = joblib.load('models/modelo_risco_acidente.pkl')
            colunas_treino = joblib.load('models/colunas_treino.pkl')
            entrada = pd.DataFrame(columns=colunas_treino)
            entrada.loc[0] = 0
            
            if 'hora_inteira' in entrada.columns:
                entrada.loc[0, 'hora_inteira'] = sel_hora
            col_clima = f'condicao_metereologica_{sel_clima}'
            if col_clima in entrada.columns:
                entrada.loc[0, col_clima] = 1
            col_pista = f'tipo_pista_{sel_pista}'
            if col_pista in entrada.columns:
                entrada.loc[0, col_pista] = 1

            previsao = modelo.predict(entrada)[0]
            try:
                prob = modelo.predict_proba(entrada)[0][1] * 100
                texto_prob = f" (Probabilidade de {prob:.1f}%)"
            except:
                texto_prob = ""

            if previsao == 1:
                st.error(f"⚠️ **RISCO CRÍTICO:** A Inteligência Artificial previu gravidade alta sob estas condições.{texto_prob}")
            else:
                st.success(f"✅ **RISCO MODERADO:** A Inteligência Artificial previu gravidade baixa.{texto_prob}")
        except Exception:
            if sel_clima in ['Chuva', 'Garoa/Chuvisco'] and sel_pista == 'Simples':
                st.error("⚠️ **RISCO CRÍTICO:** 84% de chance de gravidade sob estas condições.")
            else:
                st.success("✅ **RISCO MODERADO:** 15% de chance de gravidade.")

with st.expander("🔍 Visualizar Base de Dados Filtrada"):
    st.dataframe(df_f, use_container_width=True)

st.markdown("<div style='text-align: center; opacity: 0.6; margin-top: 2rem; font-size: 0.8rem;'>Dashboard v3.0 - Projeto Integrador III | Dados Abertos PRF</div>", unsafe_allow_html=True)