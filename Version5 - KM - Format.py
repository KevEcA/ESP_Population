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
        df = pd.read_csv(uploaded_file)  # fallback si no es tabulado

    st.write("Columnas detectadas:", df.columns.tolist())

    if "Run_Date" in df.columns:
        df["Run_Date"] = pd.to_datetime(df["Run_Date"], errors="coerce")
    if "Stop_Date" in df.columns:
        df["Stop_Date"] = pd.to_datetime(df["Stop_Date"], errors="coerce")

    # Conversión robusta de fechas
    for col in ["Run_Date", "Stop_Date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    def fix_year(date):
        if pd.notna(date) and date.year < 1970:
            return date.replace(year=date.year + 100)
        return date

    df["Run_Date"] = df["Run_Date"].apply(fix_year)
    df["Stop_Date"] = df["Stop_Date"].apply(fix_year)

    # --- Detectar rango de años disponible ---
    all_years = pd.concat([df["Run_Date"].dt.year, df["Stop_Date"].dt.year], ignore_index=True)
    all_years = all_years.dropna().astype(int)
    min_year, max_year = all_years.min(), all_years.max()
    available_years = list(range(min_year, max_year + 1))

    # --- Selector de años ---
    years = st.multiselect(texts[lang]["years"], available_years, default=available_years)
    
    
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
    max_rl_viva = int(df_viva_temp["RL_at_year"].max())
    
    # Añadir el máximo como último bin
    bins_viva.append(max_rl_viva)
    
    # --- Gráficas población viva ---
    results_viva = []
    viva_all = []
    for year in years:
        cutoff = datetime(year, 12, 31)
        active = df[(df["Run_Date"] <= cutoff) & ((df["Stop_Date"].isna()) | (df["Stop_Date"] > cutoff))].copy()
        active["RL_at_year"] = (cutoff - active["Run_Date"]).dt.days
    
        # Crear intervalos y etiquetas consistentes
        intervals = pd.cut(active["RL_at_year"], bins=bins_viva, right=False)
        labels = [str(cat) for cat in intervals.cat.categories]   # etiquetas ordenadas y consistentes
        active["RL_segment"] = pd.Categorical(intervals.astype(str), categories=labels, ordered=True)
        active["Year"] = str(year)   # forzar Year como string
    
        # Agrupar por bin y año (una fila por combinación)
        counts = active.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_viva.append(counts)
        viva_all.append(active)
    
    if results_viva:
        # Concatenar y asegurar que todas las combinaciones (segmento x año) existan
        all_counts = pd.concat(results_viva, ignore_index=True)
    
        # Definir orden de segmentos y años
        all_segments = labels
        all_years = [str(y) for y in years]
    
        # Reindexar para incluir combinaciones faltantes con Count = 0
        idx = pd.MultiIndex.from_product([all_segments, all_years], names=["RL_segment", "Year"])
        final_viva = all_counts.set_index(["RL_segment", "Year"]).reindex(idx, fill_value=0).reset_index()
    
        # Forzar tipos categóricos y ordenados (importante para que Plotly agrupe correctamente)
        final_viva["RL_segment"] = pd.Categorical(final_viva["RL_segment"], categories=all_segments, ordered=True)
        final_viva["Year"] = pd.Categorical(final_viva["Year"], categories=all_years, ordered=True)
    
        viva_final = pd.concat(viva_all, ignore_index=True)
        viva_final["RL_segment"] = pd.Categorical(viva_final["RL_segment"], categories=all_segments, ordered=True)
        viva_final["Year"] = viva_final["Year"].astype(str)
    
        col1, col2 = st.columns(2)
        with col1:
            fig_bar_viva = px.bar(final_viva, x="RL_segment", y="Count", color="Year", barmode=bar_mode_viva)
            st.plotly_chart(fig_bar_viva, use_container_width=True)
        with col2:
            fig_box_viva = px.box(viva_final, x="RL_segment", y="RL_at_year", color="Year")
            st.plotly_chart(fig_box_viva, use_container_width=True)

                
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
    df_fail_temp["RL_at_year"] = (df_fail_temp["Stop_Date"] - df_fail_temp["Run_Date"]).dt.days
    max_rl_fail = int(df_fail_temp["RL_at_year"].max()) if not df_fail_temp["RL_at_year"].isna().all() else 0
    
    # Añadir el máximo como último bin (si max > último bin definido)
    if max_rl_fail > bins_fail[-1]:
        bins_fail.append(max_rl_fail)
    else:
        # asegurar que el último bin sea al menos mayor que el anterior
        bins_fail.append(bins_fail[-1] + 1)
    
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
            # crear estructura vacía con columnas esperadas para mantener consistencia
            failed = failed.assign(RL_at_year=pd.Series(dtype="float64"))
    
        failed["RL_at_year"] = (failed["Stop_Date"] - failed["Run_Date"]).dt.days
    
        # Crear intervalos y etiquetas consistentes
        intervals = pd.cut(failed["RL_at_year"].fillna(-1), bins=bins_fail, right=False)
        labels = [str(cat) for cat in intervals.cat.categories]
        # Para valores NaN (por ejemplo filas vacías) usamos etiqueta del primer bin si corresponde
        failed["RL_segment"] = pd.Categorical(intervals.astype(str), categories=labels, ordered=True)
        failed["Year"] = str(year)
    
        # Agrupar por bin y año (una fila por combinación)
        counts = failed.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_fail.append(counts)
        fail_all.append(failed)
    
    if results_fail:
        # Concatenar y asegurar que todas las combinaciones (segmento x año) existan
        all_counts = pd.concat(results_fail, ignore_index=True)
    
        # Definir orden de segmentos y años
        all_segments = labels
        all_years = [str(y) for y in years]
    
        # Reindexar para incluir combinaciones faltantes con Count = 0
        idx = pd.MultiIndex.from_product([all_segments, all_years], names=["RL_segment", "Year"])
        final_fail = all_counts.set_index(["RL_segment", "Year"]).reindex(idx, fill_value=0).reset_index()
    
        # Forzar tipos categóricos y ordenados (importante para que Plotly agrupe correctamente)
        final_fail["RL_segment"] = pd.Categorical(final_fail["RL_segment"], categories=all_segments, ordered=True)
        final_fail["Year"] = pd.Categorical(final_fail["Year"], categories=all_years, ordered=True)
    
        # Preparar dataframe para boxplot (mantener filas originales)
        fail_final = pd.concat(fail_all, ignore_index=True)
        if "RL_segment" in fail_final.columns:
            fail_final["RL_segment"] = pd.Categorical(fail_final["RL_segment"], categories=all_segments, ordered=True)
        fail_final["Year"] = fail_final["Year"].astype(str)
    
        col3, col4 = st.columns(2)
        with col3:
            fig_bar_fail = px.bar(final_fail, x="RL_segment", y="Count", color="Year", barmode=bar_mode_fail)
            st.plotly_chart(fig_bar_fail, use_container_width=True)
        with col4:
            fig_box_fail = px.box(fail_final, x="RL_segment", y="RL_at_year", color="Year")
            st.plotly_chart(fig_box_fail, use_container_width=True)

    # """
    # # --- Validación de estados ---
    # st.write("Conteo de estados:", df["State"].value_counts(dropna=False))
    # #st.write("Ejemplo de 2026:", df[df["Run_Date"].dt.year == 2026][["Well_ID","Run_Date","Stop_Date","State"]])

    # # --- Conteo de pozos instalados en 2025 ---
    # df_2025 = df[df["Run_Date"].dt.year == 2025].copy()
    
    # # Total instalados
    # total_instalados = len(df_2025)
    
    # # Vivos (sin Stop_Date)
    # vivos = df_2025["Stop_Date"].isna().sum()
    
    # # Con Stop_Date
    # con_stop = df_2025["Stop_Date"].notna().sum()
    
    # # De los que tienen Stop_Date: cuántos son fallas (State=1) y cuántos censurados (State=0)
    # fallas = df_2025[(df_2025["Stop_Date"].notna()) & (df_2025["State"] == 1)].shape[0]
    # censurados = df_2025[(df_2025["Stop_Date"].notna()) & (df_2025["State"] == 0)].shape[0]
    
    # # Mostrar resultados
    # st.write("Pozos instalados en 2025:", total_instalados)
    # st.write("Vivos (sin Stop_Date):", vivos)
    # st.write("Con Stop_Date:", con_stop)
    # st.write("   - Fallas (State=1):", fallas)
    # st.write("   - Censurados (State=0):", censurados)
    # """

    # -------------------------------------------------------------------------------------------#
    # --- Preparar datos para KM ---
    df_km = df.copy()
    
    # Normalizar State: None → 0
    df_km["State"] = pd.to_numeric(df_km["State"], errors="coerce").fillna(0).astype(int)
    
    # Duración (si Stop_Date está vacío, se usa fecha actual)
    df_km["duration"] = (df_km["Stop_Date"].fillna(datetime.today()) - df_km["Run_Date"]).dt.days
    
    # Evento: sólo falla real (State=1), excepto si la causa es Manual off o Tbg/Csg
    df_km["event"] = df_km.apply(
        lambda row: 1 if (row["State"] == 1 and row["Cause"] not in ["Manual off", "Tbg/Csg"] and pd.notna(row["Stop_Date"]))
        else 0,
        axis=1
    )
    
    
    
       
    # --- Kaplan–Meier ---
    st.subheader(texts[lang]["km_header"])
    
    # Preparar datos
    df_km = df.copy()
    
    # Normalizar State: None → 0
    df_km["State"] = pd.to_numeric(df_km["State"], errors="coerce").fillna(0).astype(int)
    
    # Duración (si Stop_Date está vacío, se usa fecha actual)
    df_km["duration"] = (df_km["Stop_Date"].fillna(datetime.today()) - df_km["Run_Date"]).dt.days
    
    # Evento: sólo falla real (State=1), excepto si la causa es Manual off o Tbg/Csg, o si Stop_Date está vacío
    df_km["event"] = df_km.apply(
        lambda row: 1 if (row["State"] == 1 and row["Cause"] not in ["Manual off", "Tbg/Csg"] and pd.notna(row["Stop_Date"]))
        else 0,
        axis=1
    )
    
    # Checkbox global para mostrar intervalos de confianza
    show_ci = st.checkbox(texts[lang]["show_ci"], value=True)
    
    # --- Curva KM total ---
    kmf = KaplanMeierFitter()
    kmf.fit(df_km["duration"], event_observed=df_km["event"], label="Total")
    
    fig_total = go.Figure()
    fig_total.add_trace(go.Scatter(
        x=kmf.survival_function_.index,
        y=kmf.survival_function_["Total"],
        mode="lines",
        line_shape="hv",   # escalones horizontales-verticales
        name="Total"
    ))
    
    if show_ci:
        ci = kmf.confidence_interval_
        fig_total.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 0],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ))
        fig_total.add_trace(go.Scatter(
            x=ci.index,
            y=ci.iloc[:, 1],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            opacity=0.2,
            showlegend=False
        ))
    
    fig_total.update_layout(
        title=texts[lang]["km_total"],
        xaxis_title=texts[lang]["x_rl"],
        yaxis_title=texts[lang]["y_surv"]
    )
    st.plotly_chart(fig_total, use_container_width=True)
    
    # --- Curvas KM por año ---
    fig_years = go.Figure()
    kmf = KaplanMeierFitter()
    
    for year in available_years:
        subset = df_km[df_km["Run_Date"].dt.year == year]
        if len(subset) > 0:
            kmf.fit(subset["duration"], event_observed=subset["event"], label=str(year))
    
            fig_years.add_trace(go.Scatter(
                x=kmf.survival_function_.index,
                y=kmf.survival_function_[str(year)],
                mode="lines",
                line_shape="hv",   # escalones horizontales-verticales
                name=str(year)
            ))
    
            if show_ci:
                ci = kmf.confidence_interval_
                fig_years.add_trace(go.Scatter(
                    x=ci.index,
                    y=ci.iloc[:, 0],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False
                ))
                fig_years.add_trace(go.Scatter(
                    x=ci.index,
                    y=ci.iloc[:, 1],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    opacity=0.2,
                    showlegend=False
                ))
    
    fig_years.update_layout(
        title=texts[lang]["km_years"],
        xaxis_title=texts[lang]["x_rl"],
        yaxis_title=texts[lang]["y_surv"]
    )
    st.plotly_chart(fig_years, use_container_width=True)
