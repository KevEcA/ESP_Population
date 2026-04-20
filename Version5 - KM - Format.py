import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from lifelines import KaplanMeierFitter
from datetime import datetime

# --- Selector de idioma ---
lang = st.selectbox("Idioma / Language", ["ES", "EN"], index=0)

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

st.title(texts[lang]["title"])
st.markdown("<small><i>Developed by Kevin Andagoya - 2026</i></small>", unsafe_allow_html=True)

sample = """Well_ID\tRun_Date\tStop_Date\tState\tCause
ACA-025 - 12\t15-Sep-23\t\t0\t
ACAB-059 - 10\t12-Jul-22\t11-Aug-24\t1\tFail
ACA-020 - 11\t16-Jul-23\t26-Sep-24\t0\tManual off
ACA-024 - 5\t03-Feb-18\t31-Aug-23\t0\tTbg/Csg
ACAC-058 - 7\t15-Sep-23\t\t0\t
"""
st.download_button(texts[lang]["download"], sample, file_name="example.txt")

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

    # Detectar rango de años disponible
    all_years = pd.concat(
        [df["Run_Date"].dt.year.dropna(), df["Stop_Date"].dt.year.dropna()],
        ignore_index=True
    ).astype(int)
    min_year, max_year = int(all_years.min()), int(all_years.max())
    available_years = list(range(min_year, max_year + 1))

    # Selector de años
    years = st.multiselect(texts[lang]["years"], available_years, default=available_years)

    # Utilidades
    def ensure_int_list_from_input(text_input):
        parts = [p.strip() for p in text_input.split(",") if p.strip() != ""]
        nums = []
        for p in parts:
            try:
                nums.append(int(float(p)))
            except Exception:
                continue
        nums = sorted(list(dict.fromkeys(nums)))
        return nums

    # Construir edges y labels con la convención solicitada:
    # usuario: 0,300,600,900  -> bins: 0-300 ; 301-600 ; 601-900 ; >=901
    def build_edges_and_labels(user_bins, max_rl_cutoff):
        if not user_bins:
            user_bins = [0, 300, 600, 900]
        user_bins = sorted(user_bins)
        # Para implementar la convención pedida:
        # - primer bin: [0,300]  -> representaremos como left=0, right=300.5 y closed='left'
        # - segundo bin: [301,600] -> left=300.5, right=600.5
        # - tercero: [601,900] -> left=600.5, right=900.5
        # - último: >=901 -> left=900.5, right=max_rl_cutoff+0.5
        left_edge = user_bins[0] - 0.5
        internal_edges = [b + 0.5 for b in user_bins[1:]]
        final_edge = max(max_rl_cutoff, user_bins[-1]) + 0.5
        edges = [left_edge] + internal_edges + [final_edge]
        labels = []
        for i in range(len(user_bins)):
            if i == 0:
                left = user_bins[0]
                right = user_bins[1] if len(user_bins) > 1 else user_bins[0]
                labels.append(f"{left}-{right}")
            elif i < len(user_bins) - 1:
                left = user_bins[i] + 1
                right = user_bins[i + 1]
                labels.append(f"{left}-{right}")
            else:
                left = user_bins[-1] + 1
                labels.append(f">={left}")
        return edges, labels

      # ---------------------------
    # BLOQUE POBLACIÓN VIVA
    # ---------------------------

    st.subheader(texts[lang]["viva_header"])

    bar_mode_viva = st.radio(
        "Modo de visualización (población viva):",
        options=["stack", "group"],
        index=1,
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )

    bins_input_viva = st.text_input(texts[lang]["bins_viva"], "0,300,600,900", key="bins_viva_main")
    user_bins_viva = ensure_int_list_from_input(bins_input_viva)
    if not user_bins_viva:
        user_bins_viva = [0, 300, 600, 900]

    # Calcular máximo RL respecto al año máximo seleccionado (para asegurar último bin)
    if years:
        max_selected_year = max(int(y) for y in years)
    else:
        max_selected_year = max_year
    cutoff_for_max = datetime(max_selected_year, 12, 31)
    rl_at_cutoff_series = (cutoff_for_max - df["Run_Date"]).dt.days
    max_rl_cutoff = int(rl_at_cutoff_series.max(skipna=True)) if not rl_at_cutoff_series.isna().all() else user_bins_viva[-1] + 1

    # Construir edges y labels (conteo: >=901; etiqueta final: "910-X" según tu pedido)
    # Usamos edges exactos y pd.cut con right=True + include_lowest=True para la convención:
    #  - 0-300 incluye 300
    #  - 301-600 incluye 301
    #  - 601-900 incluye 601..900
    #  - último bin incluye >=901 (conteo correcto) pero lo etiquetamos como "910-X"
    edges_viva = [user_bins_viva[0], user_bins_viva[1], user_bins_viva[2], user_bins_viva[3], max_rl_cutoff]
    # Etiquetas visibles (última etiqueta mostrará 910-max_rl_cutoff)
    labels_viva = [
        f"{user_bins_viva[0]}-{user_bins_viva[1]}",
        f"{user_bins_viva[1]+1}-{user_bins_viva[2]}",
        f"{user_bins_viva[2]+1}-{user_bins_viva[3]}",
        f"910-{max_rl_cutoff}"
    ]

    # Crear IntervalIndex y mapa Interval -> label (fuente de verdad)
    # Usamos closed='right' e include_lowest en pd.cut para que 300 vaya al primer bin, 301 al segundo, etc.
    interval_index_viva = pd.IntervalIndex.from_breaks(edges_viva, closed="right")
    # Construimos label_map pero asignamos la etiqueta final manualmente al último intervalo
    label_map_viva = {}
    for i, iv in enumerate(interval_index_viva):
        if i < len(labels_viva) - 1:
            label_map_viva[iv] = labels_viva[i]
        else:
            # último intervalo: conteo real es >=901, pero etiqueta solicitada "910-X"
            label_map_viva[iv] = labels_viva[-1]

    # Recolectar datos por año
    results_viva = []
    viva_all = []
    for year in years:
        cutoff = datetime(int(year), 12, 31)
        active = df[(df["Run_Date"] <= cutoff) & ((df["Stop_Date"].isna()) | (df["Stop_Date"] > cutoff))].copy()
        active["RL_at_year"] = (cutoff - active["Run_Date"]).dt.days

        # pd.cut con IntervalIndex devuelve Interval objects; usamos include_lowest=True
        intervals = pd.cut(active["RL_at_year"], bins=interval_index_viva, right=True, include_lowest=True)
        # Mapear Interval -> label usando label_map_viva (exacto)
        active["RL_segment"] = intervals.map(label_map_viva)
        active["Year"] = str(year)

        counts = active.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_viva.append(counts)
        viva_all.append(active)

    if results_viva:
        all_counts = pd.concat(results_viva, ignore_index=True)
        # Asegurar que RL_segment sea objeto y mantener None si no asignado
        all_counts["RL_segment"] = all_counts["RL_segment"].astype(object).where(all_counts["RL_segment"].notna(), None)

        all_segments = labels_viva
        all_years = [str(y) for y in years]
        idx = pd.MultiIndex.from_product([all_segments, all_years], names=["RL_segment", "Year"])
        final_viva = all_counts.set_index(["RL_segment", "Year"]).reindex(idx, fill_value=0).reset_index()

        final_viva["RL_segment"] = pd.Categorical(final_viva["RL_segment"], categories=all_segments, ordered=True)
        final_viva["Year"] = pd.Categorical(final_viva["Year"], categories=all_years, ordered=True)

        viva_final = pd.concat(viva_all, ignore_index=True)
        viva_final["RL_segment_interval"] = pd.cut(viva_final["RL_at_year"], bins=interval_index_viva, right=True, include_lowest=True)
        viva_final["RL_segment"] = viva_final["RL_segment_interval"].map(label_map_viva)
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

    # ---------------------------
    # BLOQUE POBLACIÓN FALLADA
    # ---------------------------

    st.subheader(texts[lang]["fail_header"])

    bar_mode_fail = st.radio(
        "Modo de visualización (población fallada):",
        options=["stack", "group"],
        index=1,
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )

    bins_input_fail = st.text_input(texts[lang]["bins_fail"], "0,300,600,900", key="bins_fail_main")
    user_bins_fail = ensure_int_list_from_input(bins_input_fail)
    if not user_bins_fail:
        user_bins_fail = [0, 300, 600, 900]

    if years:
        max_selected_year = max(int(y) for y in years)
    else:
        max_selected_year = max_year
    cutoff_for_max_fail = datetime(max_selected_year, 12, 31)
    rl_at_cutoff_series_fail = (cutoff_for_max_fail - df["Run_Date"]).dt.days
    max_rl_cutoff_fail = int(rl_at_cutoff_series_fail.max(skipna=True)) if not rl_at_cutoff_series_fail.isna().all() else user_bins_fail[-1] + 1

    edges_fail, labels_fail = build_edges_and_labels(user_bins_fail, max_rl_cutoff_fail)
    interval_index_fail = pd.IntervalIndex.from_breaks(edges_fail, closed="left")
    label_map_fail = {interval_index_fail[i]: labels_fail[i] for i in range(len(interval_index_fail))}

    results_fail = []
    fail_all = []
    for year in years:
        failed = df[
            (df["Stop_Date"].dt.year == int(year)) &
            ((df["State"] == 1) | (df["Cause"] == "Tbg/Csg")) &
            (df["Cause"] != "Manual off")
        ].copy()

        if failed.empty:
            failed = pd.DataFrame(columns=df.columns.tolist())

        failed["RL_at_year"] = (failed["Stop_Date"] - failed["Run_Date"]).dt.days
        intervals_f = pd.cut(failed["RL_at_year"], bins=interval_index_fail, right=False)
        failed["RL_segment"] = intervals_f.map(label_map_fail)
        failed["Year"] = str(year)

        counts = failed.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_fail.append(counts)
        fail_all.append(failed)

    if results_fail:
        all_counts_fail = pd.concat(results_fail, ignore_index=True)
        all_counts_fail["RL_segment"] = all_counts_fail["RL_segment"].astype(object).where(all_counts_fail["RL_segment"].notna(), None)

        all_segments_fail = labels_fail
        all_years_fail = [str(y) for y in years]
        idx_fail = pd.MultiIndex.from_product([all_segments_fail, all_years_fail], names=["RL_segment", "Year"])
        final_fail = all_counts_fail.set_index(["RL_segment", "Year"]).reindex(idx_fail, fill_value=0).reset_index()

        final_fail["RL_segment"] = pd.Categorical(final_fail["RL_segment"], categories=all_segments_fail, ordered=True)
        final_fail["Year"] = pd.Categorical(final_fail["Year"], categories=all_years_fail, ordered=True)

        fail_final = pd.concat(fail_all, ignore_index=True)
        if not fail_final.empty:
            fail_final["RL_segment_interval"] = pd.cut(fail_final["RL_at_year"], bins=interval_index_fail, right=False)
            fail_final["RL_segment"] = fail_final["RL_segment_interval"].map(label_map_fail)
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

    # ---------------------------
    # BLOQUE KAPLAN–MEIER
    # ---------------------------

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

    # Filtrar según selección de años
    if years:
        df_km_filtered = df_km[df_km["Run_Date"].dt.year.isin([int(y) for y in years])].copy()
    else:
        df_km_filtered = df_km.copy()

    # KM total (sobre datos filtrados)
    kmf_total = KaplanMeierFitter()
    if len(df_km_filtered) > 0:
        kmf_total.fit(df_km_filtered["duration"], event_observed=df_km_filtered["event"], label="Total")
        fig_total = go.Figure()
        fig_total.add_trace(go.Scatter(
            x=kmf_total.survival_function_.index,
            y=kmf_total.survival_function_["Total"],
            mode="lines",
            line_shape="hv",
            name="Total",
            line=dict(width=2, color="black")
        ))
        if show_ci:
            ci = kmf_total.confidence_interval_
            fig_total.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig_total.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,0,0,0.08)", showlegend=False, hoverinfo="skip"))
        fig_total.update_layout(title=texts[lang]["km_total"], xaxis_title=texts[lang]["x_rl"], yaxis_title=texts[lang]["y_surv"])
        st.plotly_chart(fig_total, use_container_width=True)
    else:
        st.info("No hay datos para la curva Kaplan–Meier total con los años seleccionados.")

    # KM por año (solo años seleccionados)
    fig_years = go.Figure()
    kmf = KaplanMeierFitter()
    years_to_plot = [int(y) for y in years] if years else []

    for year in years_to_plot:
        subset = df_km[df_km["Run_Date"].dt.year == year]
        if len(subset) == 0:
            continue
        label = str(year)
        kmf.fit(subset["duration"], event_observed=subset["event"], label=label)
        fig_years.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[label],
            mode="lines",
            line_shape="hv",
            name=label,
            legendgroup=label,
            line=dict(width=2),
            hovertemplate="Días: %{x}<br>Survival: %{y:.3f}<extra>" + label + "</extra>"
        ))
        if show_ci:
            ci = kmf.confidence_interval_
            fig_years.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 0], mode="lines", line=dict(width=0), showlegend=False, legendgroup=label, hoverinfo="skip"))
            fig_years.add_trace(go.Scatter(x=ci.index, y=ci.iloc[:, 1], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(0,0,0,0.12)", showlegend=False, legendgroup=label, hoverinfo="skip"))

    if years_to_plot:
        fig_years.update_layout(legend=dict(traceorder="normal"))

    fig_years.update_layout(title=texts[lang]["km_years"], xaxis_title=texts[lang]["x_rl"], yaxis_title=texts[lang]["y_surv"])
    st.plotly_chart(fig_years, use_container_width=True)
