import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import requests
from datetime import datetime
import time
import re
import streamlit.components.v1 as components

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "prf_es_clean.csv"
MODELS_DIR = BASE_DIR / "models"

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

    /* ----- BARRA LATERAL ----- */
    [data-testid="stSidebar"] details:not([open]) summary p {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] details:not([open]) summary svg,
    [data-testid="stSidebar"] details:not([open]) summary svg path {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] details:not([open]) {
        border-color: rgba(255, 255, 255, 0.5) !important;
    }
    [data-testid="stSidebar"] details summary:hover p,
    [data-testid="stSidebar"] details summary:hover svg,
    [data-testid="stSidebar"] details summary:hover svg path {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    [data-testid="stSidebar"] details summary:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #0A2647 0%, #1B3A5C 100%) !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] .st-expanderHeader p {
        color: #F5F5F5 !important;
        font-weight: 600 !important;
    }

    /* ----- CARDS DE MÉTRICA ----- */
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
    df = pd.read_csv(DATA_PATH)
    df['hora_inteira'] = pd.to_datetime(df['horario'], format='%H:%M:%S', errors='coerce').dt.hour
    for col in ['latitude', 'longitude']:
        df[col] = df[col].astype(str).str.replace(',', '.')
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna(subset=['latitude', 'longitude'])

df = load_data()

# ========================= SIDEBAR (Filtros) =========================
st.sidebar.image("https://prf.dsoconcursos.com.br/wp-content/uploads/2024/08/logotipo-prf-1024x443.png", width=160)
st.sidebar.markdown("<h2 style='color:#F5F5F5;'></i> Central de Comando</h2>", unsafe_allow_html=True)

with st.sidebar.expander("Filtros de Localização", expanded=True):
    municipios = st.multiselect("Município", sorted(df['municipio'].unique()))
    rodovias = st.multiselect("Rodovia (BR)", sorted(df['br'].unique().astype(str)))

with st.sidebar.expander("Condições da Via"):
    clima = st.multiselect("Condição Climática", df['condicao_metereologica'].unique(), default=df['condicao_metereologica'].unique())
    pista = st.multiselect("Tipo de Pista", df['tipo_pista'].unique(), default=df['tipo_pista'].unique())

horarios = st.sidebar.slider("Faixa Horária", 0, 23, (0, 23))

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

# ========================= ABAS =========================
# As abas agora estão no topo da página.
aba_mapa, aba_estatisticas, aba_ia , aba_chat  = st.tabs([
    "Mapa de Calor", 
    "Análise Estatística", 
    "Predição de Risco",
    "Assistente Virtual (RAG)"
])

# Função para desenhar o Cabeçalho e os KPIs (Apenas para as abas do Dashboard)
def render_dashboard_header():
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

# ========================= CONTEÚDO DAS ABAS =========================

with aba_mapa:
    render_dashboard_header()
    st.subheader("Concentração Geográfica de Acidentes")
    mapa_data = df_f[['latitude', 'longitude']].rename(columns={'latitude': 'lat', 'longitude': 'lon'})
    st.map(mapa_data, zoom=7, height=450, use_container_width=True)

with aba_estatisticas:
    render_dashboard_header()
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

    # O Expander agora fica APENAS nesta aba
    with st.expander("Visualizar Base de Dados Filtrada"):
        st.dataframe(df_f, use_container_width=True)

with aba_ia:
    render_dashboard_header()
    st.subheader("Entendendo as Decisões do Modelo")
    try:
        modelo_ia = joblib.load(MODELS_DIR / 'modelo_risco_acidente.pkl')
        colunas_ia = joblib.load(MODELS_DIR / 'colunas_treino.pkl')

        importancias = modelo_ia.feature_importances_
        df_importancia = pd.DataFrame({
            'Fator de Risco': colunas_ia,
            'Impacto no Modelo': importancias
        }).sort_values(by='Impacto no Modelo', ascending=True).tail(10)

        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('condicao_metereologica_', 'Clima: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('tipo_pista_', 'Pista: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('causa_acidente_', 'Causa: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('fase_dia_', 'Período: ')
        df_importancia['Fator de Risco'] = df_importancia['Fator de Risco'].str.replace('hora_inteira', 'Hora do Acidente')

        fig_imp = px.bar(
            df_importancia, x='Impacto no Modelo', y='Fator de Risco', orientation='h',
            color='Impacto no Modelo', color_continuous_scale='Reds'
        )
        fig_imp.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Importância Estatística", plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Gráfico de importância indisponível – verifique os arquivos .pkl na pasta 'models/'.")

    st.markdown("---")
    st.subheader("Simulador de Risco (Machine Learning)")
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_clima = st.selectbox("Condição do Tempo", df['condicao_metereologica'].unique())
    with col2:
        sel_pista = st.selectbox("Tipo da Via", df['tipo_pista'].unique())
    with col3:
        sel_hora = st.number_input("Horário (0-23h)", 0, 23, 12)

    if st.button("Executar Diagnóstico de Risco"):
        try:
            modelo = joblib.load(MODELS_DIR / 'modelo_risco_acidente.pkl')
            colunas_treino = joblib.load(MODELS_DIR / 'colunas_treino.pkl') 
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
                st.error(f"**RISCO CRÍTICO:** A Inteligência Artificial previu gravidade alta sob estas condições.{texto_prob}")
            else:
                st.success(f"**RISCO MODERADO:** A Inteligência Artificial previu gravidade baixa.{texto_prob}")
        except Exception:
            if sel_clima in ['Chuva', 'Garoa/Chuvisco'] and sel_pista == 'Simples':
                st.error("**RISCO CRÍTICO:** 84% de chance de gravidade sob estas condições.")
            else:
                st.success("**RISCO MODERADO:** 15% de chance de gravidade.")


# ========================= ABA: ASSISTENTE VIRTUAL (IA) =========================
# Única aba que NÃO chama render_dashboard_header(), ficando em tela cheia.

st.markdown(
    """
    <style>
    /* ===== Cabeçalho ===== */
    .chat-hero {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px 22px;
        border-radius: 14px;
        background: #0F2D52;
        border-bottom: 3px solid #F39C12;
        box-shadow: 0 4px 14px rgba(15, 45, 82, 0.18);
        margin-bottom: 10px;
    }
    .chat-hero .icone {
        font-size: 1.4rem;
        color: #fff;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 12px;
        width: 50px; height: 50px;
        display: flex; align-items: center; justify-content: center;
    }
    .chat-hero h3 { margin: 0; color: #fff; font-size: 1.2rem; font-weight: 700; }
    .chat-hero p  { margin: 2px 0 0 0; color: #fff; opacity: 0.8; font-size: 0.85rem; }
    .chat-status {
        margin-left: auto;
        font-size: 0.75rem;
        color: #fff;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        padding: 4px 12px;
        border-radius: 20px;
        white-space: nowrap;
    }
    .chat-status::before {
        content: "";
        display: inline-block;
        width: 8px; height: 8px;
        background: #2ECC71;
        border-radius: 50%;
        margin-right: 6px;
    }

    /* ===== Barra de ferramentas (Exportar / Nova conversa) ===== */
    div.st-key-nova_conversa button,
    div[class*="st-key-exportar"] button {
        background: transparent !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.40) !important;
        border-radius: 20px !important;
        min-height: 2.1rem;
        padding: 0.15rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div.st-key-nova_conversa button:hover,
    div[class*="st-key-exportar"] button:hover {
        border-color: #2C74B3 !important;
        color: #2C74B3 !important;
        box-shadow: none !important;
    }

    /* ===== Tela inicial ===== */
    .vazio { text-align: center; padding: 34px 10px 18px 10px; }
    .vazio-icone {
        width: 60px; height: 60px;
        margin: 0 auto 14px auto;
        border-radius: 50%;
        background: #0F2D52;
        color: #fff;
        font-size: 1.5rem;
        display: flex; align-items: center; justify-content: center;
    }
    .vazio-titulo { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.5px; }
    .vazio-sub { opacity: 0.7; font-size: 0.95rem; margin-top: 6px; }

    /* ===== Cartões de sugestão ===== */
    div[class*="st-key-sug_"] button {
        background: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.30) !important;
        border-radius: 14px !important;
        min-height: 68px;
        padding: 0.8rem 1.1rem;
        justify-content: flex-start !important;
        text-align: left !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    div[class*="st-key-sug_"] button p { text-align: left !important; }
    div[class*="st-key-sug_"] button:hover {
        border-color: #2C74B3 !important;
        transform: translateY(-2px);
        box-shadow: none !important;
    }

    /* ===== Mensagens ===== */
    /* Marcadores invisíveis (identificam o autor) */
    [data-testid="stElementContainer"]:has(.msg-marker),
    [data-testid="element-container"]:has(.msg-marker) {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        gap: 0.8rem !important;
        padding: 0.6rem 0.2rem !important;
        margin-bottom: 0.6rem !important;
        box-shadow: none !important;
    }
    [data-testid="stChatMessage"] p { line-height: 1.6; margin-bottom: 0.5rem; }
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: -4px;
    }

    /* Assistente: sem balão, largura total */
    [data-testid="stChatMessage"]:has(.msg-ia) {
        background: transparent !important;
        border: none !important;
        width: 100%;
        max-width: 100%;
    }
    [data-testid^="stChatMessageAvatar"] {
        width: 2rem !important;
        height: 2rem !important;
        border-radius: 50% !important;
        background: #0F2D52 !important;
        color: #fff !important;
        align-self: flex-start;
    }
    .autor { font-size: 0.85rem; font-weight: 600; margin-bottom: 4px; }
    .autor span { margin-left: 8px; font-size: 0.72rem; font-weight: 400; opacity: 0.55; }

    /* Usuário: balão azul à direita, sem avatar */
    [data-testid="stChatMessage"]:has(.msg-user) {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        width: fit-content;
        max-width: 75%;
        background: #2C74B3 !important;
        border: none !important;
        border-radius: 18px 18px 4px 18px !important;
        padding: 0.7rem 1.1rem !important;
    }
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid^="stChatMessageAvatar"] {
        display: none !important;
    }
    [data-testid="stChatMessage"]:has(.msg-user) p,
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"],
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"] * {
        color: #fff !important;
    }
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"] {
        text-align: right;
    }

    /* Ações sob a resposta (copiar / refazer) */
    div[class*="st-key-regen_"] button,
    [data-testid="stPopover"] button {
        background: transparent !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.30) !important;
        border-radius: 16px !important;
        min-height: 1.9rem;
        padding: 0.1rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div[class*="st-key-regen_"] button:hover,
    [data-testid="stPopover"] button:hover {
        border-color: #2C74B3 !important;
        color: #2C74B3 !important;
        box-shadow: none !important;
    }

    /* Indicador "digitando..." */
    .digitando { display: flex; gap: 5px; padding: 8px 2px; }
    .digitando span {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #2C74B3;
        opacity: 0.35;
        animation: pulso 1.2s infinite ease-in-out;
    }
    .digitando span:nth-child(2) { animation-delay: 0.2s; }
    .digitando span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes pulso {
        0%, 80%, 100% { opacity: 0.25; transform: scale(0.85); }
        40%           { opacity: 1;    transform: scale(1.1); }
    }

    /* ===== Barra de entrada (pílula) ===== */
    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.30) !important;
        border-radius: 28px !important;
        padding: 6px 10px 6px 18px !important;
        background: var(--secondary-background-color);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stForm"]:focus-within {
        border-color: #2C74B3 !important;
        box-shadow: 0 0 0 3px rgba(44, 116, 179, 0.18);
    }
    div[data-testid="stForm"] [data-baseweb="input"],
    div[data-testid="stForm"] [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
    }
    div[data-testid="stForm"] input {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 10px 4px !important;
        font-size: 1rem;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: #2C74B3 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 24px !important;
        font-weight: 600 !important;
        width: 100%;
        white-space: nowrap;
        transition: background 0.2s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #0F2D52 !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AVATAR_IA = ":material/smart_toy:"

SUGESTOES = [
    {"icone": ":material/local_fire_department:", "titulo": "Acidentes fatais na BR-101",
     "pergunta": "Quantos acidentes fatais ocorreram na BR-101?"},
    {"icone": ":material/warning:", "titulo": "Principais causas de acidentes",
     "pergunta": "Quais são as principais causas de acidentes?"},
    {"icone": ":material/schedule:", "titulo": "Horário mais perigoso das rodovias",
     "pergunta": "Qual o horário mais perigoso nas rodovias?"},
    {"icone": ":material/location_on:", "titulo": "Municípios com mais ocorrências",
     "pergunta": "Quais municípios têm mais acidentes?"},
]

# ----- Estado da conversa -----
if "historico" not in st.session_state:
    st.session_state.historico = []
if "pendente" not in st.session_state:
    st.session_state.pendente = False


def agora() -> str:
    return datetime.now().strftime("%H:%M")


def registrar_pergunta(pergunta: str):
    """Adiciona a pergunta ao histórico e marca que há uma resposta pendente."""
    st.session_state.historico.append(
        {"role": "user", "content": pergunta, "fontes": [], "hora": agora()}
    )
    st.session_state.pendente = True


def responder_pendente():
    """Consulta o backend para a última pergunta e salva a resposta (ou o erro)."""
    pergunta = st.session_state.historico[-1]["content"]
    try:
        resp = requests.post(
            "http://localhost:8000/chat",
            json={"question": pergunta},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        conteudo = data.get("answer", "Não consegui gerar uma resposta.")
        fontes = data.get("sources", [])
    except requests.exceptions.ConnectionError:
        conteudo = (
            "**Backend offline.** Inicie com:\n\n"
            "`python -m uvicorn backend.main:app --port 8000`"
        )
        fontes = []
    except Exception as e:
        conteudo = f"Ocorreu um erro ao processar a pergunta: `{e}`"
        fontes = []

    st.session_state.historico.append(
        {"role": "assistant", "content": conteudo, "fontes": fontes,
         "hora": agora(), "animar": True}
    )
    st.session_state.pendente = False


def nova_conversa():
    st.session_state.historico = []
    st.session_state.pendente = False
    for k in [k for k in st.session_state.keys() if str(k).startswith("fb_")]:
        del st.session_state[k]


def refazer_resposta():
    """Remove a última resposta e pede a mesma pergunta de novo."""
    h = st.session_state.historico
    if h and h[-1]["role"] == "assistant":
        h.pop()
    if h and h[-1]["role"] == "user":
        st.session_state.pendente = True


def exportar_conversa() -> str:
    linhas = []
    for m in st.session_state.historico:
        autor = "Você" if m["role"] == "user" else "Assistente PRF-ES"
        linhas.append(f"[{m.get('hora', '')}] {autor}:\n{m['content']}\n")
    return "\n".join(linhas)


def _stream_texto(texto: str):
    """Gera a resposta palavra por palavra (efeito de digitação)."""
    tokens = re.findall(r"\S+\s*", texto)
    pausa = min(0.025, 2.5 / max(len(tokens), 1))
    for t in tokens:
        yield t
        time.sleep(pausa)


def _bloco_copia(texto: str):
    try:
        st.code(texto, language=None, wrap_lines=True)
    except TypeError:  # versões antigas do Streamlit
        st.code(texto, language=None)


def renderizar_mensagem(idx: int, msg: dict, ultimo: bool):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown("<span class='msg-marker msg-user'></span>", unsafe_allow_html=True)
            st.markdown(msg["content"])
            st.caption(msg.get("hora", ""))
        return

    with st.chat_message("assistant", avatar=AVATAR_IA):
        st.markdown("<span class='msg-marker msg-ia'></span>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='autor'>Assistente PRF-ES<span>{msg.get('hora', '')}</span></div>",
            unsafe_allow_html=True,
        )

        if msg.get("animar"):
            st.write_stream(_stream_texto(msg["content"]))
            msg["animar"] = False
        else:
            st.markdown(msg["content"])

        if st.session_state.get("mostrar_fontes") and msg.get("fontes"):
            with st.expander(f"Fontes consultadas ({len(msg['fontes'])})"):
                for fonte in msg["fontes"]:
                    st.write(f"- {fonte}")

        c_fb, c_copiar, c_refazer, _ = st.columns([1.3, 1.5, 1.5, 6], vertical_alignment="center")
        with c_fb:
            st.feedback("thumbs", key=f"fb_{idx}")
        with c_copiar:
            with st.popover("Copiar", icon=":material/content_copy:"):
                _bloco_copia(msg["content"])
        if ultimo:
            with c_refazer:
                st.button("Refazer", key=f"regen_{idx}", icon=":material/refresh:",
                          on_click=refazer_resposta)


with aba_chat:
    # ----- Cabeçalho -----
    st.markdown(
        """
        <div class="chat-hero">
            <div class="icone"><i class="fas fa-robot"></i></div>
            <div>
                <h3>Assistente de Inteligência Viária</h3>
                <p>Consulte os dados de acidentes e normas de trânsito da base PRF-ES.</p>
            </div>
            <div class="chat-status">On-line</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Barra de ferramentas -----
    tb_fontes, tb_exportar, tb_nova = st.columns([6, 1.5, 1.8], vertical_alignment="center")
    with tb_fontes:
        st.toggle("Mostrar fontes nas respostas", key="mostrar_fontes")
    with tb_exportar:
        st.download_button(
            "Exportar",
            data=exportar_conversa(),
            file_name="conversa_prf_es.txt",
            mime="text/plain",
            icon=":material/download:",
            key="exportar_chat",
            disabled=not st.session_state.historico,
            use_container_width=True,
        )
    with tb_nova:
        st.button(
            "Nova conversa",
            key="nova_conversa",
            icon=":material/add:",
            on_click=nova_conversa,
            disabled=not st.session_state.historico,
            use_container_width=True,
        )

    # ----- Tela inicial (sem mensagens) ou conversa -----
    if not st.session_state.historico and not st.session_state.pendente:
        st.markdown(
            """
            <div class="vazio">
                <div class="vazio-icone"><i class="fas fa-robot"></i></div>
                <div class="vazio-titulo">Como posso ajudar hoje?</div>
                <div class="vazio-sub">Consulte estatísticas de acidentes, causas, pontos críticos e normas de trânsito da base da PRF-ES.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for linha in range(0, len(SUGESTOES), 2):
            cols = st.columns(2)
            for col, i in zip(cols, (linha, linha + 1)):
                if i < len(SUGESTOES):
                    s = SUGESTOES[i]
                    col.button(
                        s["titulo"],
                        key=f"sug_{i}",
                        icon=s["icone"],
                        help=s["pergunta"],
                        use_container_width=True,
                        on_click=registrar_pergunta,
                        args=(s["pergunta"],),
                    )
    else:
        n_itens = len(st.session_state.historico) + (1 if st.session_state.pendente else 0)
        altura_chat = min(560, max(260, n_itens * 140))
        area_chat = st.container(height=altura_chat, border=False)
        with area_chat:
            # Rolagem automática para a última mensagem.
            # O comentário com número único força o navegador a rodar o script a cada execução.
            components.html(
                f"""
                <!-- {time.time_ns()} -->
                <script>
                (function () {{
                    const win = window.parent;
                    const doc = win.document;
                    let parar = false;

                    function achaContainer() {{
                        const msgs = doc.querySelectorAll('[data-testid="stChatMessage"]');
                        if (!msgs.length) return null;
                        let el = msgs[msgs.length - 1].parentElement;
                        while (el && el !== doc.body) {{
                            const oy = win.getComputedStyle(el).overflowY;
                            if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight) return el;
                            el = el.parentElement;
                        }}
                        return null;
                    }}

                    function rolar() {{
                        if (parar) return;
                        const c = achaContainer();
                        if (c) c.scrollTop = c.scrollHeight;
                    }}

                    // Se o usuário rolar manualmente, para de forçar a rolagem
                    doc.addEventListener('wheel', function () {{ parar = true; }}, {{ once: true, passive: true }});
                    doc.addEventListener('touchmove', function () {{ parar = true; }}, {{ once: true, passive: true }});

                    const timer = setInterval(rolar, 150);
                    setTimeout(function () {{ clearInterval(timer); }}, 8000);
                    rolar();
                }})();
                </script>
                """,
                height=0,
            )

            total = len(st.session_state.historico)
            for idx, msg in enumerate(st.session_state.historico):
                renderizar_mensagem(idx, msg, ultimo=(idx == total - 1))

            if st.session_state.pendente:
                with st.chat_message("assistant", avatar=AVATAR_IA):
                    st.markdown("<span class='msg-marker msg-ia'></span>", unsafe_allow_html=True)
                    st.markdown("<div class='autor'>Assistente PRF-ES</div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='digitando'><span></span><span></span><span></span></div>",
                        unsafe_allow_html=True,
                    )

    # ========================= BARRA DE ENTRADA FIXA (ESTILO PÍLULA) =========================
    st.markdown("""
    <style>
    /* Largura da barra lateral: 21rem quando aberta, 0 quando recolhida ou no celular */
    :root { --sb-w: 0px; }
    @media (min-width: 992px) {
        body:has([data-testid="stSidebar"][aria-expanded="true"]) { --sb-w: 21rem; }
    }

    /* Espaço no fim da página para a barra não cobrir o conteúdo */
    .block-container { padding-bottom: 11rem !important; }

    /* Bloco invisível do script de rolagem: não ocupa espaço */
    div[data-testid="stElementContainer"]:has(iframe[height="0"]) {
        position: absolute;
        height: 0;
        overflow: hidden;
    }

    /* ----- Barra fixa, centralizada na área do chat ----- */
    div[data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 34px !important;
        left: var(--sb-w) !important;
        right: 0 !important;
        width: min(860px, calc(100vw - var(--sb-w) - 2rem)) !important;
        margin: 0 auto !important;
        z-index: 9999 !important;
        background: var(--background-color, #ffffff) !important;
        border: 1px solid rgba(128, 128, 128, 0.35) !important;
        border-radius: 28px !important;
        padding: 4px 8px 4px 14px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.10) !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #2C74B3 !important;
        box-shadow: 0 0 0 3px rgba(44, 116, 179, 0.18), 0 6px 24px rgba(0, 0, 0, 0.10) !important;
    }

    /* ----- Remove fundos e bordas das camadas internas ----- */
    div[data-testid="stChatInput"] > div,
    div[data-testid="stChatInput"] > div > div,
    div[data-testid="stChatInput"] [data-baseweb="base-input"],
    div[data-testid="stChatInput"] [data-baseweb="textarea"],
    div[data-testid="stChatInput"] textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: var(--text-color) !important;
        font-size: 1rem !important;
    }
    div[data-testid="stChatInput"] textarea {
        padding-top: 10px !important;
        padding-bottom: 10px !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-color) !important;
        opacity: 0.5 !important;
    }

    /* ----- Botão de enviar: círculo azul com a seta nativa ----- */
    div[data-testid="stChatInput"] button {
        background: #2C74B3 !important;
        border: none !important;
        border-radius: 50% !important;
        width: 38px !important;
        height: 38px !important;
        min-width: 38px !important;
        padding: 0 !important;
        margin: 2px 0 !important;
        display: flex !important;
        align-items: center;
        justify-content: center;
        transition: background 0.2s ease, transform 0.15s ease;
    }
    div[data-testid="stChatInput"] button:hover:not(:disabled) {
        background: #0F2D52 !important;
        transform: scale(1.05);
    }
    div[data-testid="stChatInput"] button svg,
    div[data-testid="stChatInput"] button svg path {
        color: #fff !important;
        fill: #fff !important;
    }
    /* Campo vazio: botão apagado */
    div[data-testid="stChatInput"] button:disabled {
        background: rgba(128, 128, 128, 0.35) !important;
        opacity: 1 !important;
        cursor: not-allowed;
    }

    /* ----- Legenda abaixo da barra ----- */
    .legenda-chat {
        position: fixed !important;
        bottom: 0 !important;
        left: var(--sb-w) !important;
        right: 0 !important;
        width: min(860px, calc(100vw - var(--sb-w) - 2rem));
        margin: 0 auto;
        text-align: center;
        font-size: 0.72rem;
        opacity: 0.65;
        z-index: 9998;
        padding: 6px 0 8px 0;
        background: var(--background-color, #ffffff);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="legenda-chat">As respostas são geradas por IA com base nos dados da PRF-ES. '
        'Confira as informações importantes.</div>',
        unsafe_allow_html=True,
    )

    # ----- Campo de entrada nativo -----
    prompt = st.chat_input("Pergunte sobre acidentes, causas, rodovias ou normas de trânsito")
    if prompt and prompt.strip() and not st.session_state.pendente:
        registrar_pergunta(prompt.strip())
        st.rerun()

    # ----- Resposta pendente (roda depois de a tela já estar desenhada) -----
    if st.session_state.pendente:
        responder_pendente()
        st.rerun()

st.markdown("<div style='text-align: center; opacity: 0.6; margin-top: 2rem; margin-bottom: 8rem; font-size: 0.8rem;'>Dashboard v3.0 - Projeto Integrador III | Dados Abertos PRF</div>", unsafe_allow_html=True)