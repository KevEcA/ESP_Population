import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from datetime import datetime

# --- Selector de idioma (dropdown sutil) ---
lang = st.selectbox("Idioma / Language", ["ES", "EN"], index=0)

# --- Diccionario de traducciones ---
texts = {
    "ES": {
        "title": "Análisis de bombas ESP",
        "download": "Descargar ejemplo de archivo",
        "upload": "Sube tu archivo .txt",
        "years": "Selecciona años a analizar",
        "bins_viva": "Definir segmentos / rangos de estudio (población viva)",
        "bins_fail": "Definir segmentos / rangos de estudio (población fallada/censurada)",
        "viva_header": "Cantidad de bombas vivas por rangos y año / Distribución de RL (población viva)",
        "fail_header": "Cantidad de bombas falladas/censuradas por rango y año / Distribución de RL (población fallada/censurada)",
        "km_header": "Kaplan–Meier",
        "km_total": "Curva Kaplan–Meier Total",
        "km_years": "Curvas Kaplan–Meier por año de arranque",
        "show_ci": "Mostrar intervalos de confianza",
        "x_rl": "Días de operación (RL)",
        "y_surv": "Probabilidad de supervivencia"
    },
    "EN": {
        "title": "ESP Pump Analysis",
        "download": "Download sample file",
        "upload": "Upload your .txt file",
        "years": "Select years to analyze",
        "bins_viva": "Define bins (active population)",
        "bins_fail": "Define bins (failed/censored population)",
        "viva_header": "Number of active pumps per quartile and year / RL distribution (active population)",
        "fail_header": "Number of failed/censored pumps per quartile and year / RL distribution (failed/censored population)",
        "km_header": "Kaplan–Meier",
        "km_total": "Kaplan–Meier Curve (Total)",
        "km_years": "Kaplan–Meier Curves by start year",
        "show_ci": "Show confidence intervals",
        "x_rl": "Operating days (RL)",
        "y_surv": "Survival probability"
    }
}

# --- Título principal ---
st.title(texts[lang]["title"])

# --- Firma en letras pequeñas ---
st.markdown(
    "<small><i>Developed by Kevin Andagoya - 2026</i></small>",
    unsafe_allow_html=True
)

# --- Enlace de ayuda debajo del título ---
st.markdown(
    '<p style="font-size:12px;">'
    '<a href="https://keveca.github.io/ESP_Population/tutorial.html" target="_blank" style="color:#007BFF; text-decoration:none;">HELP?</a>'
    '</p>',
    unsafe_allow_html=True
)

# --- Ejemplo descargable ---
sample = """Well_ID\tRun_Date\tStop_Date\tState\tCause
ACA-025 - 12\t15-Sep-23\t\t0\t
ACAB-059 - 10\t12-Jul-22\t11-Aug-24\t1\tFail
ACA-020 - 11\t16-Jul-23\t26-Sep-24\t0\tManual off
ACA-024 - 5\t03-Feb-18\t31-Aug-23\t0\tTbg/Csg
ACAC-058 - 7\t15-Sep-23\t\t0\t
"""
st.download_button(texts[lang]["download"], sample, file_name="example.txt")

# --- Cargar archivo ---
uploaded_file = st.file_uploader(texts[lang]["upload"], type=["txt","csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, sep="\t")
    except Exception:
        df = pd.read_csv(uploaded_file)

    st.write("Columnas detectadas:", df.columns.tolist())

    # Conversión robusta de fechas
    for col in ["Run_Date", "Stop_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    def fix_year(date):
        if pd.notna(date) and date.year < 1970:
            return date.replace(year=date.year + 100)
        return date

    if "Run_Date" in df.columns:
        df["Run_Date"] = df["Run_Date"].apply(fix_year)
    if "Stop_Date" in df.columns:
        df["Stop_Date"] = df["Stop_Date"].apply(fix_year)

    # --- Detectar rango de años disponible ---
    all_years = pd.concat(
        [df["Run_Date"].dt.year.dropna(), df["Stop_Date"].dt.year.dropna()],
        ignore_index=True
    ).astype(int)
    min_year, max_year = int(all_years.min()), int(all_years.max())
    available_years = list(range(min_year, max_year + 1))

    # --- Selector de años ---
    years = st.multiselect(texts[lang]["years"], available_years, default=available_years)

    # ---------------------------
    # --- BLOQUE POBLACIÓN VIVA ---
    # ---------------------------

    # --- Titulo de gráficas población viva ---
    st.subheader(texts[lang]["viva_header"])

    # --- Selector de modo de barras ---
    bar_mode_viva = st.radio(
        "Modo de visualización (población viva):",
        options=["stack", "group"],
        index=1,  # por defecto "group"
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )

    # --- Definir bins población viva ---
    bins_input_viva = st.text_input(texts[lang]["bins_viva"], "0,300,600,900")
    bins_viva = [int(x) for x in bins_input_viva.split(",")]

    # Calcular el máximo RL disponible en la población viva
    df_viva_temp = df.copy()
    df_viva_temp["RL_at_year"] = (df_viva_temp["Stop_Date"].fillna(datetime.today()) - df_viva_temp["Run_Date"]).dt.days
    max_rl_viva = int(df_viva_temp["RL_at_year"].max()) if not df_viva_temp["RL_at_year"].isna().all() else bins_viva[-1] + 1

    # Asegurar que el último bin sea mayor que el anterior
    if max_rl_viva <= bins_viva[-1]:
        max_rl_viva = bins_viva[-1] + 1

    # Añadir el máximo como último bin
    bins_viva.append(int(max_rl_viva))

    # Crear IntervalIndex y etiquetas consistentes (fuente de verdad)
    interval_index_viva = pd.IntervalIndex.from_breaks(bins_viva, closed="left")
    labels_viva = [str(iv) for iv in interval_index_viva]

    # --- Gráficas población viva ---
    results_viva = []
    viva_all = []
    for year in years:
        cutoff = datetime(year, 12, 31)
        active = df[(df["Run_Date"] <= cutoff) & ((df["Stop_Date"].isna()) | (df["Stop_Date"] > cutoff))].copy()
        active["RL_at_year"] = (cutoff - active["Run_Date"]).dt.days

        # Asignar RL_segment usando los mismos bins (evita inconsistencias)
        intervals = pd.cut(active["RL_at_year"], bins=bins_viva, right=False)
        active["RL_segment"] = intervals.astype(str)
        active["Year"] = str(year)

        # Agrupar por bin y año (una fila por combinación)
        counts = active.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_viva.append(counts)
        viva_all.append(active)

    if results_viva:
        # Concatenar y normalizar etiquetas para que coincidan con labels_viva
        all_counts = pd.concat(results_viva, ignore_index=True)
        all_counts["RL_segment"] = all_counts["RL_segment"].astype(str)

        # Normalizar formatos: mapear cada etiqueta a la etiqueta estándar en labels_viva
        def normalize_label(s, labels):
            for lab in labels:
                # comparar límites iniciales para evitar diferencias .0 vs int
                if s.startswith(lab.split(",")[0].strip("[")):
                    return lab
            return s

        all_counts["RL_segment"] = all_counts["RL_segment"].apply(lambda s: normalize_label(s, labels_viva))

        # Definir orden y años
        all_segments = labels_viva
        all_years = [str(y) for y in years]

        # Reindexar para incluir combinaciones faltantes con Count = 0
        idx = pd.MultiIndex.from_product([all_segments, all_years], names=["RL_segment", "Year"])
        final_viva = all_counts.set_index(["RL_segment", "Year"]).reindex(idx, fill_value=0).reset_index()

        # Forzar tipos categóricos y ordenados (clave para orden correcto)
        final_viva["RL_segment"] = pd.Categorical(final_viva["RL_segment"], categories=all_segments, ordered=True)
        final_viva["Year"] = pd.Categorical(final_viva["Year"], categories=all_years, ordered=True)

        # Preparar dataframe original para boxplot: usar las mismas etiquetas y tipo categorical
        viva_final = pd.concat(viva_all, ignore_index=True)
        viva_final["RL_segment"] = pd.cut(viva_final["RL_at_year"], bins=bins_viva, right=False)
        viva_final["RL_segment"] = viva_final["RL_segment"].astype(str)
        viva_final["RL_segment"] = viva_final["RL_segment"].apply(lambda s: normalize_label(s, labels_viva))
        viva_final["RL_segment"] = pd.Categorical(viva_final["RL_segment"], categories=all_segments, ordered=True)
        viva_final["Year"] = viva_final["Year"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar_viva = px.bar(final_viva, x="RL_segment", y="Count", color="Year", barmode=bar_mode_viva)
            st.plotly_chart(fig_bar_viva, use_container_width=True)
        with col2:
            fig_box_viva = px.box(viva_final, x="RL_segment", y="RL_at_year", color="Year")
            fig_box_viva.update_xaxes(categoryorder="array", categoryarray=all_segments)
            st.plotly_chart(fig_box_viva, use_container_width=True)

    # -----------------------------
    # --- BLOQUE POBLACIÓN FALLADA ---
    # -----------------------------

    # --- Titulo de gráficas población fallada ---
    st.subheader(texts[lang]["fail_header"])

    # --- Selector de modo de barras ---
    bar_mode_fail = st.radio(
        "Modo de visualización (población fallada):",
        options=["stack", "group"],
        index=1,  # por defecto "group"
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )

    # --- Definir bins población fallada ---
    bins_input_fail = st.text_input(texts[lang]["bins_fail"], "0,300,600,900")
    bins_fail = [int(x) for x in bins_input_fail.split(",")]

    # Calcular el máximo RL disponible en la población fallada
    df_fail_temp = df[df["Stop_Date"].notna()].copy()
    if not df_fail_temp.empty:
        df_fail_temp["RL_at_year"] = (df_fail_temp["Stop_Date"] - df_fail_temp["Run_Date"]).dt.days
        max_rl_fail = int(df_fail_temp["RL_at_year"].max())
    else:
        max_rl_fail = bins_fail[-1] + 1

    # Asegurar que el último bin sea mayor que el anterior
    if max_rl_fail <= bins_fail[-1]:
        max_rl_fail = bins_fail[-1] + 1

    # Añadir el máximo como último bin
    bins_fail.append(int(max_rl_fail))

    # Crear IntervalIndex y etiquetas consistentes para fallada
    interval_index_fail = pd.IntervalIndex.from_breaks(bins_fail, closed="left")
    labels_fail = [str(iv) for iv in interval_index_fail]

    # --- Gráficas población fallada/censurada ---
    results_fail = []
    fail_all = []
    for year in years:
        failed = df[
            (df["Stop_Date"].dt.year == year) &
            ((df["State"] == 1) | (df["Cause"] == "Tbg/Csg")) &
            (df["Cause"] != "Manual off")
        ].copy()

        if failed.empty:
            # mantener estructura vacía con columnas esperadas
            failed = pd.DataFrame(columns=df.columns.tolist())

        failed["RL_at_year"] = (failed["Stop_Date"] - failed["Run_Date"]).dt.days
        failed["RL_segment"] = pd.cut(failed["RL_at_year"], bins=bins_fail, right=False).astype(str)
        failed["Year"] = str(year)

        counts = failed.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_fail.append(counts)
        fail_all.append(failed)

    if results_fail:
        all_counts_fail = pd.concat(results_fail, ignore_index=True)
        all_counts_fail["RL_segment"] = all_counts_fail["RL_segment"].astype(str)

        # Normalizar etiquetas a labels_fail
        def normalize_label_fail(s, labels):
            for lab in labels:
                if s.startswith(lab.split(",")[0].strip("[")):
                    return lab
            return s

        all_counts_fail["RL_segment"] = all_counts_fail["RL_segment"].apply(lambda s: normalize_label_fail(s, labels_fail))

        all_segments_fail = labels_fail
        all_years_fail = [str(y) for y in years]

        idx_fail = pd.MultiIndex.from_product([all_segments_fail, all_years_fail], names=["RL_segment", "Year"])
        final_fail = all_counts_fail.set_index(["RL_segment", "Year"]).reindex(idx_fail, fill_value=0).reset_index()

        final_fail["RL_segment"] = pd.Categorical(final_fail["RL_segment"], categories=all_segments_fail, ordered=True)
        final_fail["Year"] = pd.Categorical(final_fail["Year"], categories=all_years_fail, ordered=True)

        fail_final = pd.concat(fail_all, ignore_index=True)
        if not fail_final.empty:
            fail_final["RL_segment"] = pd.cut(fail_final["RL_at_year"], bins=bins_fail, right=False).astype(str)
            fail_final["RL_segment"] = fail_final["RL_segment"].apply(lambda s: normalize_label_fail(s, labels_fail))
            fail_final["RL_segment"] = pd.Categorical(fail_final["RL_segment"], categories=all_segments_fail, ordered=True)
            fail_final["Year"] = fail_final["Year"].astype(str)

        col3, col4 = st.columns(2)
        with col3:
            fig_bar_fail = px.bar(final_fail, x="RL_segment", y="Count", color="Year", barmode=bar_mode_fail)
            st.plotly_chart(fig_bar_fail, use_container_width=True)
        with col4:
            fig_box_fail = px.box(fail_final, x="RL_segment", y="RL_at_year", color="Year")
            fig_box_fail.update_xaxes(categoryorder="array", categoryarray=all_segments_fail)
            st.plotly_chart(fig_box_fail, use_container_width=True)

    # -----------------------------
    # --- BLOQUE KAPLAN–MEIER ---
    # -----------------------------

    st.subheader(texts[lang]["km_header"])

    df_km = df.copy()
    df_km["State"] = pd.to_numeric(df_km["State"], errors="coerce").fillna(0).astype(int)
    df_km["duration"] = (df_km["Stop_Date"].fillna(datetime.today()) - df_km["Run_Date"]).dt.days
    df_km["event"] = df_km.apply(
        lambda row: 1 if (row["State"] == 1 and row.get("Cause") not in ["Manual off", "Tbg/Csg"] and pd.notna(row["Stop_Date"]))
        else 0,
        axis=1
    )

    show_ci = st.checkbox(texts[lang]["show_ci"], value=True)

    # Curva KM total
    kmf = KaplanMeierFitter()
    kmf.fit(df_km["duration"], event_observed=df_km["event"], label="Total")

    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_["Total"],
        mode="lines",
        line_shape="hv",
        name="Total"
    ))

    if show_ci:
        ci = kmf.confidence_interval_
        fig_total.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
        fig_total.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", opacity=0.2, showlegend=False))

    fig_total.update_layout(title=texts[lang]["km_total"], xaxis_title=texts[lang]["x_rl"], yaxis_title=texts[lang]["y_surv"])
    st.plotly_chart(fig_total, use_container_width=True)

    # Curvas KM por año
    fig_years = go.Figure()
    for year in available_years:
        subset = df_km[df_km["Run_Date"].dt.year == year]
        if len(subset) > 0:
            kmf.fit(subset["duration"], event_observed=subset["event"], label=str(year))
            fig_years.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[str(year)],
                mode="lines",
                line_shape="hv",
                name=str(year)
            ))
            if show_ci:
                ci = kmf.confidence_interval_
                fig_years.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False))
                fig_years.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", opacity=0.2, showlegend=False))

    fig_years.update_layout(title=texts[lang]["km_years"], xaxis_title=texts[lang]["x_rl"], yaxis_title=texts[lang]["y_surv"])
    st.plotly_chart(fig_years, use_container_width=True)
