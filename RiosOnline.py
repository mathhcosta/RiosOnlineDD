# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import streamlit.components.v1 as components
import plotly.graph_objects as go
import os
import requests
from folium.plugins import LocateControl
from streamlit_js_eval import get_geolocation
import sqlite3
from datetime import datetime

# ================= CONFIG =================
st.set_page_config(
    page_title="Sala de Situação – Rios Online",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #e9f3ff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="
        text-align:center;
        padding:4px 0px;
        margin-top:-15px;
        margin-bottom:8px;
        border-bottom:1px solid #d9d9d9;
    ">
        <span style="
            font-size:22px;
            font-weight:600;
            letter-spacing:0.5px;
            color:#1f4e79;
        ">
            Sala de Situação – Rios Online
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


col0, col1, col2, col3, col4 = st.columns([2,3,3,3,2])

with col2:
    st.image("static/logos/logo.png", width=260)

st.markdown("---")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================= FUNÇÃO LEITURA =================
BANCO_1 = os.path.join(BASE_DIR, "bancos", "hidrologia1.db")
BANCO_2 = os.path.join(BASE_DIR, "bancos", "hidrologia2.db")

LISTA_BANCOS = [BANCO_1, BANCO_2]

# ================= COLUNAS =================
def carregar_dados_estacao(codigo):
    """
    Procura a estação nos dois bancos.
    Retorna os dados do banco onde a estação for encontrada.
    """

    for banco in LISTA_BANCOS:

        if not os.path.exists(banco):
            continue

        conn = sqlite3.connect(banco)

        try:
            # ---------------------------
            # Série diária (cotas)
            # ---------------------------
            df_cotas = pd.read_sql_query("""
                SELECT data, cota
                FROM cotas
                WHERE codigo_estacao = ?
                ORDER BY data
            """, conn, params=(codigo,))

            # Se não encontrou essa estação nesse banco
            if df_cotas.empty:
                conn.close()
                continue

            df_cotas["data"] = pd.to_datetime(df_cotas["data"])

            # ---------------------------
            # Estatísticas anuais
            # ---------------------------
            df_anuais = pd.read_sql_query("""
                SELECT *
                FROM dados_estatisticos_anuais
                WHERE codigo_estacao = ?
                ORDER BY ano
            """, conn, params=(codigo,))

            # ---------------------------
            # Frequência mensal
            # ---------------------------
            df_freq = pd.read_sql_query("""
                SELECT *
                FROM frequencia_meses
                WHERE codigo_estacao = ?
            """, conn, params=(codigo,))

            # ---------------------------
            # Climatologia diária
            # ---------------------------
            df_clima = pd.read_sql_query("""
                SELECT *
                FROM maxmin
                WHERE codigo_estacao = ?
                ORDER BY mes, dia
            """, conn, params=(codigo,))

            conn.close()

            return df_cotas, df_anuais, df_freq, df_clima

        except Exception as e:
            conn.close()
            continue

    # Se não encontrou em nenhum banco
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()



# ================= GRÁFICOS =================
def grafico_hidrograma(df_clima, df_cotas):
    """
    Hidrôgrama diário com eixo X estético apenas pelos meses (Jan-Dec)
    Faixa histórica diária e série atual do ano atual.
    Linhas limpas, sem marcadores, valores arredondados.
    """
    import calendar
    import numpy as np

    # Usar ano fictício bissexto 2000 para evitar problemas com 29/02
    ano_ficticio = 2000

    df_clima["data_plot"] = pd.to_datetime(
        ano_ficticio*10000 + df_clima["mes"]*100 + df_clima["dia"],
        format="%Y%m%d",
        errors="coerce"  # datas inválidas (ex: 30/02) viram NaT
    )

    # Mesclar coluna dia/mes para hover
    df_clima["dia_mes"] = df_clima["data_plot"].dt.strftime("%d/%m")

    # Ano atual
    ano_atual = pd.Timestamp.now().year
    df_atual = df_cotas[df_cotas["data"].dt.year == ano_atual].copy()

    if not df_atual.empty:
        # Criar coluna data fictícia para alinhar climatologia
        df_atual["data_plot"] = pd.to_datetime(
            "2000-" +
            df_atual["data"].dt.month.astype(str).str.zfill(2) + "-" +
            df_atual["data"].dt.day.astype(str).str.zfill(2)
        )
        df_atual["cota"] = df_atual["cota"].round(0)
    else:
        df_atual = pd.DataFrame({"data_plot": [], "cota": []})

    # Arredondar climatologia
    df_clima["cota_max"] = df_clima["cota_max"].round(0)
    df_clima["cota_min"] = df_clima["cota_min"].round(0)
    df_clima["media"] = df_clima["media"].round(0)

    fig = go.Figure()

    # Faixa histórica
    fig.add_trace(go.Scatter(
        x=df_clima["data_plot"],
        y=df_clima["cota_max"],
        line=dict(width=0),
        showlegend=False,
        mode="lines"
    ))

    fig.add_trace(go.Scatter(
        x=df_clima["data_plot"],
        y=df_clima["cota_min"],
        fill="tonexty",
        fillcolor="rgba(0,100,255,0.15)",
        line=dict(width=0),
        name="Faixa Histórica",
        mode="lines"
    ))

    # Média histórica
    fig.add_trace(go.Scatter(
        x=df_clima["data_plot"],
        y=df_clima["media"],
        line=dict(color="green", width=2),
        name="Média Histórica",
        mode="lines"
    ))

    # Série atual
    if not df_atual.empty:
        fig.add_trace(go.Scatter(
            x=df_atual["data_plot"],
            y=df_atual["cota"],
            line=dict(color="red", width=2),
            name=f"Ano {ano_atual}",
            mode="lines"
        ))

    # Eixo X fixo apenas nos meses (Jan, Feb, Mar...)
    meses_nome = [calendar.month_abbr[i] for i in range(1, 13)]
    # Posição aproximada para cada mês: primeiro dia do mês no ano 2000
    meses_pos = pd.to_datetime(["2000-%02d-01" % i for i in range(1,13)])

    fig.update_layout(
        height=350,
        xaxis=dict(
            tickmode="array",
            tickvals=meses_pos,
            ticktext=meses_nome,
            title="Mês"
        ),
        yaxis_title="Cota (cm)",
        hovermode="x unified"
    )

    return fig

def obter_pais_por_gps(lat, lon):
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
        headers = {"User-Agent": "RiosOnlineApp"}

        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()

        return data["address"].get("country", "").lower()

    except:
        return None


def grafico_variabilidade(df_anuais):

    ano_atual = datetime.now().year

    # 1️⃣ Remover ano corrente
    df = df_anuais[df_anuais["ano"] < ano_atual].copy()

    # 2️⃣ Remover anos com dados faltantes
    df = df.dropna(subset=["variabilidade"])

    # (opcional extra segurança)
    df = df[df["variabilidade"] != 0]

    # 3️⃣ Ordenar por ano decrescente e pegar os 10 últimos válidos
    df_10 = df.sort_values("ano", ascending=False).head(10)

    # 4️⃣ Reordenar crescente para o gráfico ficar cronológico
    df_10 = df_10.sort_values("ano")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_10["ano"],
        y=df_10["variabilidade"],
        mode="lines+markers"
    ))

    fig.update_layout(
        height=350,
        xaxis_title="Ano",
        yaxis_title="Variabilidade"
    )

    return fig


def obter_cota_atual(df_cotas):
    """
    Retorna a última cota do ano atual e a data correspondente.
    Se não houver dados no ano atual, retorna None, None.
    """
    ano_atual = pd.Timestamp.now().year  # ex: 2026
    df_ano = df_cotas[df_cotas["data"].dt.year == ano_atual]

    if df_ano.empty:
        return None, None

    # pega o último registro do ano atual
    ultima_linha = df_ano.iloc[-1]
    return ultima_linha["cota"], ultima_linha["data"]


def grafico_frequencia(df_freq, coluna, titulo):
    """
    Gera gráfico de pizza para frequência mensal.
    Filtra dados nulos e garante que valores são numéricos.
    Ordena meses de 1 a 12 para consistência.
    """
    import calendar
    import numpy as np

    if df_freq.empty or coluna not in df_freq.columns:
        # DataFrame vazio ou coluna inexistente
        return go.Figure()

    # Garantir que os valores são numéricos
    df_freq = df_freq.copy()
    df_freq[coluna] = pd.to_numeric(df_freq[coluna], errors="coerce").fillna(0)

    # Garantir que os meses estão no formato 1-12 e ordenar
    df_freq["mes"] = pd.to_numeric(df_freq["mes"], errors="coerce").fillna(0).astype(int)
    df_freq = df_freq.sort_values("mes")

    # Substituir 0 por NaN para não aparecer no gráfico
    df_freq.loc[df_freq[coluna] <= 0, coluna] = np.nan

    # Converter meses para nomes
    df_freq["mes_nome"] = df_freq["mes"].apply(lambda x: calendar.month_abbr[x] if 1 <= x <= 12 else "")

    # Remover entradas inválidas
    df_freq = df_freq.dropna(subset=[coluna, "mes_nome"])

    fig = go.Figure()

    fig.add_trace(go.Pie(
        labels=df_freq["mes_nome"],
        values=df_freq[coluna],
        hole=0.35,
        textinfo="percent+label",
        sort=False
    ))

    fig.update_layout(
        title=titulo,
        height=350
    )

    return fig


import plotly.graph_objects as go

def tabela_eventos_figura(df_anuais, tipo="cheia"):
    """
    Retorna um gráfico de tabela Plotly com os 5 principais eventos:
    Colunas: Data | Cota (cm)
    Fundo branco, visual limpo.
    """
    if tipo == "cheia":
        df = df_anuais.sort_values("cota_maxima", ascending=False).head(5).copy()
        df["Data"] = pd.to_datetime(
            df["ano"].astype(str) + "-" +
            df["mes_cota_maxima"].astype(str).str.zfill(2) + "-" +
            df["dia_cota_maxima"].astype(str).str.zfill(2)
        ).dt.strftime("%d/%m/%Y")
        df["Cota (cm)"] = df["cota_maxima"].round(0).astype(int)

    else:
        df = df_anuais.sort_values("cota_minima", ascending=True).head(5).copy()
        df["Data"] = pd.to_datetime(
            df["ano"].astype(str) + "-" +
            df["mes_cota_minima"].astype(str).str.zfill(2) + "-" +
            df["dia_cota_minima"].astype(str).str.zfill(2)
        ).dt.strftime("%d/%m/%Y")
        df["Cota (cm)"] = df["cota_minima"].round(0).astype(int)

    # Apenas duas colunas de interesse
    df_plot = df[["Data", "Cota (cm)"]]

    # Criar figura da tabela
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_plot.columns),
            fill_color="white",
            font=dict(color="black", size=14),
            align="center"
        ),
        cells=dict(
            values=[df_plot["Data"], df_plot["Cota (cm)"]],
            fill_color="white",
            font=dict(color="black", size=12),
            align="center"
        )
    )])

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=10, b=10)
    )

    return fig

# ================= ESTAÇÕES =================
def carregar_estacoes(pasta_estacoes):
    arquivo = os.path.join(pasta_estacoes, "estacoes.xlsx")

    if not os.path.exists(arquivo):
        st.error(f"Arquivo de estações não encontrado: {arquivo}")
        st.stop()

    df = pd.read_excel(arquivo)

    estacoes = []
    for _, row in df.iterrows():
        lat = float(str(row["lat"]).replace(",", "."))
        lon = float(str(row["lon"]).replace(",", "."))

        estacoes.append({
            "codigo": str(row["codigo"]),
            "nome": row["nome"],
            "coords": [lat, lon],
            "tipo": row.get("tipo", ""),
            "pais": [
                p.strip().lower()
                for p in str(row.get("pais", "")).split(",")
            ]

        })


    return estacoes


# ================= INPUT =================

pasta_estacoes = os.path.join(BASE_DIR, "estacoes")
if "codigo_estacao" not in st.session_state:
    st.session_state["codigo_estacao"] = None

estacoes = carregar_estacoes(pasta_estacoes)

# ================= MAPAS =================

col_mapa1, col_mapa2 = st.columns(2)

# ================= GPS AUTOMÁTICO =================
if "gps_carregado" not in st.session_state:
    st.session_state["gps_carregado"] = False
    st.session_state["pais_gps"] = None
    st.session_state["lat_user"] = None
    st.session_state["lon_user"] = None

# Captura automática ao abrir
if not st.session_state["gps_carregado"]:
    loc = get_geolocation()

    if loc:
        lat = loc["coords"]["latitude"]
        lon = loc["coords"]["longitude"]

        st.session_state["lat_user"] = lat
        st.session_state["lon_user"] = lon
        st.session_state["pais_gps"] = obter_pais_por_gps(lat, lon)
        st.session_state["gps_carregado"] = True

# ================= LISTA DE PAÍSES =================
lista_paises = sorted(
    set(p for e in estacoes for p in e["pais"] if p)
)

# Define país padrão automaticamente
pais_gps = st.session_state["pais_gps"]

if pais_gps and pais_gps in lista_paises:
    indice_padrao = lista_paises.index(pais_gps)
else:
    indice_padrao = 0

# ================= SELECTBOX =================
pais_selecionado = st.selectbox(
    "🌎 Filtrar estações por país",
    lista_paises,
    index=indice_padrao
)

# ================= MAPA ESTAÇÕES =================
with col_mapa1:
    st.markdown(
        """
        <div style="
            font-size:16px;
            font-weight:600;
            margin-bottom:8px;
            color:#1f4e79;
        ">
            Estações fluviais
        </div>
        """,
        unsafe_allow_html=True
    )

    # Centro padrão
    lat_centro = -3.5
    lon_centro = -60
    zoom_mapa = 5

    # Se GPS capturado
    if st.session_state["lat_user"]:
        lat_centro = st.session_state["lat_user"]
        lon_centro = st.session_state["lon_user"]
        zoom_mapa = 8

    mapa = folium.Map(
        location=[lat_centro, lon_centro],
        zoom_start=zoom_mapa,
        tiles="OpenStreetMap",
        control_scale=True
    )

    LocateControl(auto_start=False).add_to(mapa)

    for e in estacoes:
        if pais_selecionado.lower() not in e["pais"]:
            continue

        folium.Marker(
            location=e["coords"],
            tooltip=e["nome"],
            icon=folium.Icon(color="blue", icon="map-pin", prefix="fa")
        ).add_to(mapa)

    retorno = st_folium(mapa, height=400, use_container_width=True)

# ================= MAPA WINDY =================
with col_mapa2:
    st.markdown(
        """
        <div style="
            font-size:16px;
            font-weight:600;
            margin-bottom:8px;
            color:#1f4e79;
        ">
            🌬️ Condições atmosféricas (Windy)
        </div>
        """,
        unsafe_allow_html=True
    )

    components.html(
        """
        <iframe width="100%" height="500"
        src="https://embed.windy.com/embed.html?type=map&location=coordinates&metricRain=mm&metricTemp=°C&metricWind=km/h&zoom=5&overlay=wind&product=ecmwf&level=surface&lat=-7.014&lon=-59.985&detailLat=-3.075&detailLon=-59.985&detail=true&message=true"
        frameborder="0"></iframe>
        """,
        height=400
    )


# ================= SELEÇÃO ESTAÇÃO =================
if retorno and retorno.get("last_object_clicked_tooltip"):
    nome = retorno["last_object_clicked_tooltip"]
    for e in estacoes:
        if e["nome"] == nome:
            st.session_state["codigo_estacao"] = e["codigo"]

codigo = st.session_state.get("codigo_estacao")

if codigo is None:
    st.info("Clique em uma estação no mapa para visualizar a Sala de Situação.")
    st.stop()

df_cotas, df_anuais, df_freq, df_clima = carregar_dados_estacao(codigo)

# ================= SALA DE SITUAÇÃO =================
c1, c2 = st.columns(2)
with c1:

    if st.session_state.get("codigo_estacao"):

        # Buscar estação selecionada
        estacao = next(
            (e for e in estacoes
             if e["codigo"] == st.session_state["codigo_estacao"]),
            None
        )

        if estacao:

            # Obter última cota
            cota_atual, data_cota = obter_cota_atual(df_cotas)

            # Formatar data
            data_txt = ""
            if data_cota is not None and pd.notna(data_cota):
                data_txt = f" em {pd.to_datetime(data_cota).strftime('%d/%m/%Y')}"

            # Caso exista valor válido
            if cota_atual is not None and pd.notna(cota_atual):

                st.markdown(
                    f"""
                    <div style="
                        background-color:#e6f2ff;
                        padding:12px 16px;
                        border-radius:10px;
                        margin-bottom:12px;
                        font-size:18px;
                        font-weight:600;
                    ">
                        📍 <b>{estacao['nome']}</b> — {estacao['codigo']}  
                        <br>
                        🌊 Cota do último registro: <b>{int(cota_atual)} cm</b>{data_txt}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                
                st.markdown(
                    f"""
                    <div style="
                        background-color:#e6f2ff;
                        padding:12px 16px;
                        border-radius:10px;
                        margin-bottom:12px;
                        font-size:18px;
                        font-weight:600;
                    ">
                        📍 <b>{estacao['nome']}</b> — {estacao['codigo']}  
                        <br>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.warning("Não há dados recentes disponíveis para esta estação.")

        else:
            st.error("Estação não encontrada.")

    else:
        st.info("Selecione uma estação no mapa para visualizar a Sala de Situação.")


c1, c2 = st.columns(2)
with c1:
    st.markdown(" Hidrograma de evolução anual")
    st.plotly_chart(grafico_hidrograma(df_clima, df_cotas), use_container_width=True)

with c2:
    st.markdown(" Variabilidade decadal Hmax-Hmin")
    st.plotly_chart(grafico_variabilidade(df_anuais), use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    st.markdown("Últimos 5 eventos de cheia")
    st.plotly_chart(tabela_eventos_figura(df_anuais, tipo="cheia"), use_container_width=True)

with c4:
    st.markdown("Últimos 5 eventos de seca")
    st.plotly_chart(tabela_eventos_figura(df_anuais, tipo="seca"), use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    st.plotly_chart(
        grafico_frequencia(df_freq, "freq_inicio_vazante", "Frequência de ocorrência de máximas"),
        use_container_width=True
    )
    
with c6:
    st.plotly_chart(
        grafico_frequencia(df_freq, "freq_final_vazante", "Frequência de ocorrência de mínimas"),
        use_container_width=True
    )
    




