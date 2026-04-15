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
        "bins_viva": "Definir bins (población viva)",
        "bins_fail": "Definir bins (población fallada/censurada)",
        "viva_header": "Cantidad de bombas vivas por cuartil y año / Distribución de RL (población viva)",
        "fail_header": "Cantidad de bombas falladas/censuradas por cuartil y año / Distribución de RL (población fallada/censurada)",
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
    '<a href="https://tuusuario.github.io/ESP_Population/tutorial.html" target="_blank" style="color:#007BFF; text-decoration:none;">HELP?</a>'
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

    # --- Definir bins ---
    bins_input_viva = st.text_input(texts[lang]["bins_viva"], "0,300,600,900")
    bins_viva = [int(x) for x in bins_input_viva.split(",")]
    bins_viva.append(999999)

    bins_input_fail = st.text_input(texts[lang]["bins_fail"], "0,300,600,900")
    bins_fail = [int(x) for x in bins_input_fail.split(",")]
    bins_fail.append(999999)

    # --- Gráficas población viva ---
    results_viva = []
    viva_all = []
    for year in years:
        cutoff = datetime(year, 12, 31)
        active = df[(df["Run_Date"] <= cutoff) & ((df["Stop_Date"].isna()) | (df["Stop_Date"] > cutoff))].copy()
        active["RL_at_year"] = (cutoff - active["Run_Date"]).dt.days
        intervals = pd.cut(active["RL_at_year"], bins=bins_viva, right=False)
        categories = [str(cat) for cat in intervals.cat.categories]
        active["RL_segment"] = pd.Categorical(intervals.astype(str), categories=categories, ordered=True)
        counts = active.groupby("RL_segment").size().reset_index(name="Count")
        counts["Year"] = year
        results_viva.append(counts)
        active["Year"] = year
        viva_all.append(active)

    if results_viva:
        final_viva = pd.concat(results_viva)
        viva_final = pd.concat(viva_all)
        st.subheader(texts[lang]["viva_header"])
        col1, col2 = st.columns(2)
        with col1:
            fig_bar_viva = px.bar(final_viva, x="RL_segment", y="Count", color="Year", barmode="group")
            st.plotly_chart(fig_bar_viva, use_container_width=True)
        with col2:
            fig_box_viva = px.box(viva_final, x="RL_segment", y="RL_at_year", color="Year")
            st.plotly_chart(fig_box_viva, use_container_width=True)

    # --- Gráficas población fallada/censurada ---
    results_fail = []
    fail_all = []
    for year in years:
        failed = df[
            (df["Stop_Date"].dt.year == year) &
            ((df["State"] == 1) | (df["Cause"] == "Tbg/Csg")) &
            (df["Cause"] != "Manual off")
        ].copy()
        failed["RL_at_year"] = (failed["Stop_Date"] - failed["Run_Date"]).dt.days
        intervals = pd.cut(failed["RL_at_year"], bins=bins_fail, right=False)
        categories = [str(cat) for cat in intervals.cat.categories]
        failed["RL_segment"] = pd.Categorical(intervals.astype(str), categories=categories, ordered=True)
        counts = failed.groupby("RL_segment").size().reset_index(name="Count")
        counts["Year"] = year
        results_fail.append(counts)
        failed["Year"] = year
        fail_all.append(failed)

    if results_fail:
        final_fail = pd.concat(results_fail)
        fail_final = pd.concat(fail_all)
        st.subheader(texts[lang]["fail_header"])
        col3, col4 = st.columns(2)
        with col3:
            fig_bar_fail = px.bar(final_fail, x="RL_segment", y="Count", color="Year", barmode="group")
            st.plotly_chart(fig_bar_fail, use_container_width=True)
        with col4:
            fig_box_fail = px.box(fail_final, x="RL_segment", y="RL_at_year", color="Year")
            st.plotly_chart(fig_box_fail, use_container_width=True)

        # --- Kaplan–Meier ---
        st.subheader(texts[lang]["km_header"])

        # Preparar datos
        df_km = df.copy()
        df_km["duration"] = (df_km["Stop_Date"].fillna(datetime.today()) - df_km["Run_Date"]).dt.days
        df_km["event"] = ((df_km["State"] == 1) | (df_km["Cause"] == "Tbg/Csg")).astype(int)

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

                # Curva principal
                fig_years.add_trace(go.Scatter(
                    x=kmf.survival_function_.index,
                    y=kmf.survival_function_[str(year)],
                    mode="lines",
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
