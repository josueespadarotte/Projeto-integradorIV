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

    /* ----- BARRA LATERAL (Sem bugar os selects) ----- */
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
aba_mapa, aba_estatisticas, aba_ia , aba_chat  = st.tabs([
    "📍 Mapa de Calor", 
    "📊 Análise Estatística", 
    "🤖 Predição de Risco",
    "💬 Assistente Virtual (RAG)"
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
        modelo_ia = joblib.load(MODELS_DIR / 'modelo_risco_acidente.pkl')
        colunas_ia = joblib.load(MODELS_DIR / 'colunas_treino.pkl')

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
    
# ========================= ABA: ASSISTENTE VIRTUAL (IA) =========================

st.markdown(
    """
    <style>
    /* ===== Banner (igual ao atual) ===== */
    .chat-hero {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 18px 22px;
        border-radius: 14px;
        background: #0F2D52;
        border-bottom: 3px solid #F39C12;
        color: #fff;
        box-shadow: 0 4px 14px rgba(15, 45, 82, 0.18);
        margin-bottom: 14px;
    }
    .chat-hero .icone {
        font-size: 1.8rem;
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 12px;
        width: 54px; height: 54px;
        display: flex; align-items: center; justify-content: center;
    }
    .chat-hero h3 { margin: 0; color: #fff; font-size: 1.25rem; font-weight: 700; }
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
        box-shadow: 0 0 6px #2ECC71;
    }

    /* ===== Sugestões (igual ao atual) ===== */
    .sugestoes-label {
        font-size: 0.8rem;
        font-weight: 600;
        opacity: 0.7;
        margin: 6px 0 4px 2px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[class*="st-key-sug_"] button {
        background: transparent !important;
        color: var(--text-color) !important;
        border: 1.5px solid #2C74B3 !important;
        border-radius: 30px !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: background 0.2s ease, color 0.2s ease;
    }
    div[class*="st-key-sug_"] button:hover {
        background: #2C74B3 !important;
        color: #fff !important;
        box-shadow: none !important;
    }

    /* ===== Barra de entrada (igual ao atual) ===== */
    div[data-testid="stForm"] {
        border: 1px solid rgba(128, 128, 128, 0.25) !important;
        border-radius: 18px !important;
        padding: 10px 14px !important;
        background: var(--secondary-background-color);
    }
    div[data-testid="stForm"] input {
        border-radius: 20px !important;
        border: 2px solid #2C74B3 !important;
        padding: 10px 16px !important;
    }
    div[data-testid="stForm"] input:focus {
        box-shadow: 0 0 0 3px rgba(44, 116, 179, 0.25) !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: #2C74B3 !important;
        color: #fff !important;
        border: none !important;
        border-radius: 30px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: background 0.2s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover {
        background: #0F2D52 !important;
        box-shadow: none !important;
    }

    /* ===== Botão Limpar (igual ao atual) ===== */
    div.st-key-limpar_chat button {
        background: transparent !important;
        color: var(--text-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.45) !important;
        border-radius: 30px !important;
        min-height: 2.2rem;
        padding: 0.2rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 500 !important;
        box-shadow: none !important;
    }
    div.st-key-limpar_chat button:hover {
        border-color: #D62828 !important;
        color: #D62828 !important;
        box-shadow: none !important;
    }

    /* =====================================================
       NOVO: CONVERSA NO ESTILO DE APP DE MENSAGENS
       ===================================================== */

    /* Marcadores invisíveis (identificam o autor da mensagem) */
    [data-testid="stElementContainer"]:has(.msg-marker),
    [data-testid="element-container"]:has(.msg-marker) {
        display: none !important;
    }

    /* Balão base */
    [data-testid="stChatMessage"] {
        gap: 0.75rem !important;
        padding: 0.75rem 1rem !important;
        margin-bottom: 0.7rem !important;
        border-radius: 18px !important;
        width: fit-content;
        max-width: 80%;
        box-shadow: none !important;
    }
    [data-testid="stChatMessage"] p { margin-bottom: 0.3rem; line-height: 1.5; }
    [data-testid="stChatMessage"] [data-testid="stCaptionContainer"] {
        margin-top: -4px;
        font-size: 0.7rem;
        opacity: 0.7;
    }

    /* Assistente: esquerda, neutro */
    [data-testid="stChatMessage"]:has(.msg-ia) {
        background: var(--secondary-background-color) !important;
        border: 1px solid rgba(128, 128, 128, 0.20) !important;
        border-bottom-left-radius: 4px !important;
        min-width: 280px;
        max-width: 88%;
    }

    /* Usuário: direita, azul sólido */
    [data-testid="stChatMessage"]:has(.msg-user) {
        flex-direction: row-reverse !important;
        margin-left: auto !important;
        background: #2C74B3 !important;
        border: 1px solid #2C74B3 !important;
        border-bottom-right-radius: 4px !important;
    }
    [data-testid="stChatMessage"]:has(.msg-user) p,
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"],
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"] * {
        color: #fff !important;
    }
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid="stCaptionContainer"] {
        text-align: right;
    }

    /* Avatares redondos */
    [data-testid^="stChatMessageAvatar"] {
        width: 2rem !important;
        height: 2rem !important;
        border-radius: 50% !important;
        background: #0F2D52 !important;
        color: #fff !important;
        align-self: flex-start;
    }
    [data-testid="stChatMessage"]:has(.msg-user) [data-testid^="stChatMessageAvatar"] {
        background: rgba(255, 255, 255, 0.22) !important;
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
    </style>
    """,
    unsafe_allow_html=True,
)

AVATAR_IA = ":material/support_agent:"
AVATAR_USER = ":material/person:"

SUGESTOES = [
    "Quantos acidentes fatais ocorreram na BR-101?",
    "Quais são as principais causas de acidentes?",
    "Qual o horário mais perigoso nas rodovias?",
]


def agora() -> str:
    return datetime.now().strftime("%H:%M")


def nova_boas_vindas() -> dict:
    return {
        "role": "assistant",
        "content": (
            "Olá! Sou o assistente virtual da PRF-ES. 👋\n\n"
            "Posso responder sobre **acidentes em rodovias**, **causas**, "
            "**pontos críticos** e **normas de trânsito**. Como posso ajudar?"
        ),
        "fontes": [],
        "hora": agora(),
    }


def registrar_pergunta(pergunta: str):
    """Adiciona a pergunta ao histórico e marca que há uma resposta pendente."""
    st.session_state.messages.append(
        {"role": "user", "content": pergunta, "fontes": [], "hora": agora()}
    )
    st.session_state.pendente = True


def responder_pendente():
    """Consulta o backend para a última pergunta e salva a resposta (ou o erro)."""
    pergunta = st.session_state.messages[-1]["content"]
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
            "⚠️ **Backend offline.** Inicie com:\n\n"
            "`python -m uvicorn backend.main:app --reload --port 8000`"
        )
        fontes = []
    except Exception as e:
        conteudo = f"❌ Ocorreu um erro ao processar a pergunta: `{e}`"
        fontes = []

    st.session_state.messages.append(
        {"role": "assistant", "content": conteudo, "fontes": fontes, "hora": agora()}
    )
    st.session_state.pendente = False


def renderizar_mensagem(msg: dict):
    eh_user = msg["role"] == "user"
    marcador = "msg-user" if eh_user else "msg-ia"
    with st.chat_message(msg["role"], avatar=AVATAR_USER if eh_user else AVATAR_IA):
        st.markdown(f"<span class='msg-marker {marcador}'></span>", unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("hora"):
            st.caption(msg["hora"])
        if msg.get("fontes"):
            with st.expander(f"📚 Fontes consultadas ({len(msg['fontes'])})"):
                for fonte in msg["fontes"]:
                    st.write(f"- {fonte}")


if "messages" not in st.session_state:
    st.session_state.messages = [nova_boas_vindas()]
if "pendente" not in st.session_state:
    st.session_state.pendente = False


with aba_chat:
    # ----- Banner (igual ao atual) -----
    st.markdown(
        """
        <div class="chat-hero">
            <div class="icone">🤖</div>
            <div>
                <h3>Assistente de Inteligência Viária</h3>
                <p>Consulte os dados de acidentes e normas de trânsito da base PRF-ES.</p>
            </div>
            <div class="chat-status">On-line</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Sugestões (só no início da conversa) -----
    if len(st.session_state.messages) <= 1 and not st.session_state.pendente:
        st.markdown("<div class='sugestoes-label'>💡 Sugestões de perguntas</div>", unsafe_allow_html=True)
        cols_sug = st.columns(len(SUGESTOES))
        for i, sugestao in enumerate(SUGESTOES):
            if cols_sug[i].button(sugestao, key=f"sug_{i}", use_container_width=True):
                registrar_pergunta(sugestao)
                st.rerun()

    # ----- Área de mensagens (altura dinâmica + scroll) -----
    n_itens = len(st.session_state.messages) + (1 if st.session_state.pendente else 0)
    altura_chat = min(520, max(130, n_itens * 105))
    area_chat = st.container(height=altura_chat, border=False)
    with area_chat:
        for msg in st.session_state.messages:
            renderizar_mensagem(msg)

        if st.session_state.pendente:
            with st.chat_message("assistant", avatar=AVATAR_IA):
                st.markdown("<span class='msg-marker msg-ia'></span>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='digitando'><span></span><span></span><span></span></div>",
                    unsafe_allow_html=True,
                )

    # ----- Barra de entrada (igual ao atual) -----
    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_enviar = st.columns([6, 1])
        with col_input:
            prompt = st.text_input(
                "Digite sua pergunta",
                placeholder="Ex: Quantos acidentes com vítimas fatais ocorreram na BR-101?",
                label_visibility="collapsed",
            )
        with col_enviar:
            submit_button = st.form_submit_button("Enviar ➤")

    # ----- Rodapé (igual ao atual) -----
    col_info, col_limpar = st.columns([5, 1], vertical_alignment="center")
    with col_info:
        st.caption("As respostas são geradas por IA com base nos dados da PRF-ES. Confira as fontes sempre que possível.")
    with col_limpar:
        if st.button("Limpar", key="limpar_chat", use_container_width=True):
            st.session_state.messages = [nova_boas_vindas()]
            st.session_state.pendente = False
            st.rerun()

    # ----- Envio -----
    if submit_button:
        if not prompt or not prompt.strip():
            st.toast("Digite uma pergunta para continuar.", icon="⚠️")
        elif not st.session_state.pendente:
            registrar_pergunta(prompt.strip())
            st.rerun()

    # ----- Resposta pendente (roda depois de a tela já estar desenhada) -----
    if st.session_state.pendente:
        responder_pendente()
        st.rerun()

st.markdown("<div style='text-align: center; opacity: 0.6; margin-top: 2rem; font-size: 0.8rem;'>Dashboard v3.0 - Projeto Integrador III | Dados Abertos PRF</div>", unsafe_allow_html=True)