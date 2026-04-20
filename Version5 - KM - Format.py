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
    # BLOQUE POBLACIÓN VIVA (REEMPLAZO: bins globales, último bin 901-en adelante)
    # ---------------------------
    
    import calendar
    from datetime import datetime
    
    def last_day_of_month(dt):
        if pd.isna(dt):
            return None
        y, m = dt.year, dt.month
        return datetime(y, m, calendar.monthrange(y, m)[1])
    
    st.subheader(texts[lang]["viva_header"])
    
    bar_mode_viva = st.radio(
        "Modo de visualización (población viva):",
        options=["stack", "group"],
        index=1,
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )
    
    # Bins de usuario (mantener tu helper)
    bins_input_viva = st.text_input(texts[lang]["bins_viva"], "0,300,600,900", key="bins_viva_main")
    user_bins_viva = ensure_int_list_from_input(bins_input_viva)
    if not user_bins_viva:
        user_bins_viva = [0, 300, 600, 900]
    
    # --- Determinar cutoff global: último día del mes más reciente presente en la data ---
    # Tomamos la última Run_Date disponible en todo el dataset
    last_run = df["Run_Date"].max()
    if pd.isna(last_run):
        # fallback a 31-dic del último año si no hay Run_Date
        global_cutoff = datetime(int(df["Run_Date"].dt.year.max()), 12, 31)
    else:
        global_cutoff = last_day_of_month(last_run)
    
    # Calcular RL_at_cutoff global (para dimensionar el último bin)
    rl_at_global = (global_cutoff - df["Run_Date"]).dt.days
    global_max_rl = int(rl_at_global.max(skipna=True)) if not rl_at_global.isna().all() else user_bins_viva[-1] + 1
    
    # Forzar que el último bin empiece en 901 (según tu requerimiento)
    last_bin_left = 901
    # Si el usuario puso un primer bin distinto a 0, respetamos los primeros límites y solo forzamos el último
    # Construimos edges globales: [user_bins_viva[0], user_bins_viva[1], user_bins_viva[2], user_bins_viva[3], global_max_rl]
    raw_edges = [user_bins_viva[0], user_bins_viva[1], user_bins_viva[2], user_bins_viva[3], global_max_rl]
    
    # Limpieza y garantía de crecimiento estricto
    edges_global = sorted(list(dict.fromkeys([int(e) for e in raw_edges])))
    # Asegurar que el penúltimo sea al menos 900 y el último >= last_bin_left
    if edges_global[-2] < 900:
        edges_global[-2] = 900
    if edges_global[-1] < last_bin_left:
        edges_global[-1] = last_bin_left if last_bin_left > edges_global[-2] else edges_global[-2] + 1
    # Si el último coincide con el anterior, empujarlo
    if edges_global[-1] <= edges_global[-2]:
        edges_global[-1] = edges_global[-2] + 1
  #####  -----------------
    # st.write("Edges globales usados para bins (viva):", edges_global)
    
    # Construir labels globales (última etiqueta representará 901-en adelante)
    labels_global = [
        f"{edges_global[0]}-{edges_global[1]}",
        f"{edges_global[1]+1}-{edges_global[2]}",
        f"{edges_global[2]+1}-{edges_global[3]}",
        f"901-{edges_global[-1]}"
    ]
    # st.write("Labels globales:", labels_global)
    
    # Crear IntervalIndex global (closed='right' para que 300 vaya al primer bin)
    try:
        interval_index_global = pd.IntervalIndex.from_breaks(edges_global, closed="right")
    except Exception:
        # ajuste defensivo
        edges_global = list(edges_global)
        for i in range(1, len(edges_global)):
            if edges_global[i] <= edges_global[i-1]:
                edges_global[i] = edges_global[i-1] + 1
        interval_index_global = pd.IntervalIndex.from_breaks(edges_global, closed="right")
        # st.write("Edges ajustados:", edges_global)
    ####------------------
    
    # Map Interval -> label (global)
    n_intervals = len(interval_index_global)
    if n_intervals != len(labels_global):
        # reconstruir labels desde intervals si hay mismatch
        labels_global = []
        for i in range(n_intervals):
            left = int(interval_index_global[i].left)
            right = int(interval_index_global[i].right)
            if i < n_intervals - 1:
                labels_global.append(f"{left}-{right}")
            else:
                labels_global.append(f"901-{right}")
    label_map_global = {interval_index_global[i]: labels_global[i] for i in range(len(interval_index_global))}
    
    # --- Recolectar datos por año usando el mismo interval_index_global ---
    results_viva = []
    viva_all = []
    for year in years:
        cutoff = None
        year = int(year)
        # regla: si es el año máximo presente en la data, usar último día del mes disponible en ese año
        data_max_year = int(df["Run_Date"].dt.year.max())
        if year == data_max_year:
            max_run_in_year = df[df["Run_Date"].dt.year == year]["Run_Date"].max()
            cutoff = last_day_of_month(max_run_in_year) if pd.notna(max_run_in_year) else datetime(year, 12, 31)
        else:
            cutoff = datetime(year, 12, 31)
    
        active = df[(df["Run_Date"] <= cutoff) & ((df["Stop_Date"].isna()) | (df["Stop_Date"] > cutoff))].copy()
        active["RL_at_year"] = (cutoff - active["Run_Date"]).dt.days
    
        # Asignar bins usando el interval_index_global (mismo para todos los años)
        intervals = pd.cut(active["RL_at_year"], bins=interval_index_global, right=True, include_lowest=True)
        active["RL_segment"] = intervals.map(label_map_global)
        active["Year"] = str(year)
    
        counts = active.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_viva.append(counts)
        viva_all.append(active)
    
    # Consolidar resultados (igual que antes, pero con labels_global)
    if results_viva:
        all_counts = pd.concat(results_viva, ignore_index=True)
        all_counts["RL_segment"] = all_counts["RL_segment"].astype(object).where(all_counts["RL_segment"].notna(), None)
    
        all_years = [str(y) for y in years]
        idx = pd.MultiIndex.from_product([labels_global, all_years], names=["RL_segment", "Year"])
        final_viva = all_counts.set_index(["RL_segment", "Year"]).reindex(idx, fill_value=0).reset_index()
    
        final_viva["RL_segment"] = pd.Categorical(final_viva["RL_segment"], categories=labels_global, ordered=True)
        final_viva["Year"] = pd.Categorical(final_viva["Year"], categories=all_years, ordered=True)
    
        viva_final = pd.concat(viva_all, ignore_index=True)
        viva_final["RL_segment_interval"] = pd.cut(viva_final["RL_at_year"], bins=interval_index_global, right=True, include_lowest=True)
        viva_final["RL_segment"] = viva_final["RL_segment_interval"].map(label_map_global)
        viva_final["RL_segment"] = pd.Categorical(viva_final["RL_segment"], categories=labels_global, ordered=True)
        viva_final["Year"] = viva_final["Year"].astype(str)
    
        col1, col2 = st.columns(2)
        with col1:
            fig_bar_viva = px.bar(final_viva, x="RL_segment", y="Count", color="Year", barmode=bar_mode_viva)
            st.plotly_chart(fig_bar_viva, use_container_width=True)
        with col2:
            fig_box_viva = px.box(viva_final, x="RL_segment", y="RL_at_year", color="Year")
            fig_box_viva.update_xaxes(categoryorder="array", categoryarray=labels_global)
            st.plotly_chart(fig_box_viva, use_container_width=True)
    
    # Validación rápida (muestra totales)
    st.write("Total activos (suma por años en final_viva):", int(final_viva["Count"].sum()))


    
    # ---------------------------
    # BLOQUE POBLACIÓN FALLADA (COMPLETO: último bin etiquetado como 901-X, X = máximo RL entre falladas)
    # ---------------------------
    
    import calendar
    from datetime import datetime
    
    st.subheader(texts[lang]["fail_header"])
    
    bar_mode_fail = st.radio(
        "Modo de visualización (población fallada):",
        options=["stack", "group"],
        index=1,
        format_func=lambda x: "Apilado" if x == "stack" else "Lado a lado"
    )
    
    # Leer bins de usuario (fallback a defaults)
    bins_input_fail = st.text_input(texts[lang]["bins_fail"], "0,300,600,900", key="bins_fail_main")
    user_bins_fail = ensure_int_list_from_input(bins_input_fail)
    if not user_bins_fail:
        user_bins_fail = [0, 300, 600, 900]
    
    # --- 1) Calcular máximo RL entre filas que tienen Stop_Date (solo población fallada) ---
    failed_with_stop = df[df["Stop_Date"].notna()].copy()
    if not failed_with_stop.empty:
        failed_with_stop["RL_at_stop"] = (failed_with_stop["Stop_Date"] - failed_with_stop["Run_Date"]).dt.days
        # Tomar máximo entero no negativo
        max_rl_fail = int(failed_with_stop["RL_at_stop"].max(skipna=True))
        if max_rl_fail < 0:
            max_rl_fail = user_bins_fail[-1] + 1
    else:
        # fallback razonable si no hay Stop_Date en todo el dataset
        max_rl_fail = user_bins_fail[-1] + 1
    
    # Diagnóstico mínimo para validar
    st.write("Máximo RL entre filas con Stop_Date (max_rl_fail):", int(max_rl_fail))
    
    # --- 2) Construir edges usando ese máximo (y forzar último bin inicio en 901) ---
    raw_edges_fail = [user_bins_fail[0], user_bins_fail[1], user_bins_fail[2], user_bins_fail[3], max_rl_fail]
    
    # Normalizar edges: enteros, orden, únicos
    edges_fail = sorted(list(dict.fromkeys([int(e) for e in raw_edges_fail])))
    
    # Forzar que el penúltimo sea al menos 900 y el último >= 901
    if len(edges_fail) >= 2:
        if edges_fail[-2] < 900:
            edges_fail[-2] = 900
        if edges_fail[-1] < 901:
            edges_fail[-1] = max(901, edges_fail[-2] + 1)
    # Asegurar crecimiento estricto
    if edges_fail[-1] <= edges_fail[-2]:
        edges_fail[-1] = edges_fail[-2] + 1
    
    st.write("Edges finales usados para población fallada:", edges_fail)
    
    # --- 3) Construir labels y IntervalIndex (closed='right' para convención: 300 incluido en primer bin) ---
    # Etiqueta final como "901-X" donde X = edges_fail[-1] (o max_rl_fail si prefieres)
    last_right = int(edges_fail[-1])
    labels_fail = [
        f"{edges_fail[0]}-{edges_fail[1]}",
        f"{edges_fail[1]+1}-{edges_fail[2]}",
        f"{edges_fail[2]+1}-{edges_fail[3]}",
        f"901-{last_right}"
    ]
    
    # Crear IntervalIndex con manejo de errores por solapamiento
    try:
        interval_index_fail = pd.IntervalIndex.from_breaks(edges_fail, closed="right")
    except Exception:
        # ajuste defensivo si hay igualdad en breaks
        edges_fail = list(edges_fail)
        for i in range(1, len(edges_fail)):
            if edges_fail[i] <= edges_fail[i-1]:
                edges_fail[i] = edges_fail[i-1] + 1
        interval_index_fail = pd.IntervalIndex.from_breaks(edges_fail, closed="right")
        st.write("Edges ajustados (fallada):", edges_fail)
        # reconstruir etiqueta final por si cambió
        last_right = int(edges_fail[-1])
        labels_fail[-1] = f"901-{last_right}"
    
    # Asegurar que labels y intervals coincidan; reconstruir labels si hace falta
    n_intervals_fail = len(interval_index_fail)
    if n_intervals_fail != len(labels_fail):
        labels_fail = []
        for i in range(n_intervals_fail):
            left = int(interval_index_fail[i].left)
            right = int(interval_index_fail[i].right)
            if i < n_intervals_fail - 1:
                labels_fail.append(f"{left}-{right}")
            else:
                labels_fail.append(f"901-{right}")
    st.write("Labels (fallada) usados:", labels_fail)
    
    # Map Interval -> label
    label_map_fail = {interval_index_fail[i]: labels_fail[i] for i in range(len(interval_index_fail))}
    
    # --- 4) Procesamiento por año (falladas/censuradas) ---
    results_fail = []
    fail_all = []
    
    for year in years:
        year = int(year)
    
        # Filtrar eventos de parada ocurridos en el año
        failed = df[
            (df["Stop_Date"].dt.year == year) &
            ((df["State"] == 1) | (df["Cause"] == "Tbg/Csg")) &
            (df["Cause"] != "Manual off")
        ].copy()
    
        if failed.empty:
            failed = pd.DataFrame(columns=df.columns.tolist())
    
        # Calcular RL al evento (Stop_Date - Run_Date)
        failed["RL_at_year"] = (failed["Stop_Date"] - failed["Run_Date"]).dt.days
    
        # Asignar bins usando el interval_index_fail (global para falladas)
        intervals_f = pd.cut(failed["RL_at_year"], bins=interval_index_fail, right=True, include_lowest=True)
        failed["RL_segment"] = intervals_f.map(label_map_fail)
        failed["Year"] = str(year)
    
        counts = failed.groupby(["RL_segment", "Year"]).size().reset_index(name="Count")
        results_fail.append(counts)
        fail_all.append(failed)
    
    # --- 5) Consolidar resultados y graficar ---
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
            fail_final["RL_segment_interval"] = pd.cut(fail_final["RL_at_year"], bins=interval_index_fail, right=True, include_lowest=True)
            fail_final["RL_segment"] = fail_final["RL_segment_interval"].map(label_map_fail)
            fail_final["RL_segment"] = pd.Categorical(fail_final["RL_segment"], categories=all_segments_fail, ordered=True)
            fail_final["Year"] = fail_final["Year"].astype(str)
    
        col3, col4 = st.columns(2)
        with col3:
            fig_bar_fail = px.bar(final_fail, x="RL_segment", y="Count", color="Year", barmode=bar_mode_fail)
            st.plotly_chart(fig_bar_fail, use_container_width=True)
        with col4:
            if not fail_final.empty:
                fig_box_fail = px.box(fail_final, x="RL_segment", y="RL_at_year", color="Year")
                fig_box_fail.update_xaxes(categoryorder="array", categoryarray=all_segments_fail)
                st.plotly_chart(fig_box_fail, use_container_width=True)
            else:
                st.info("No hay datos de falladas para los años seleccionados.")
    
    # # --- 6) Comprobación opcional: mostrar top RL_at_stop para validar el X final ---
    # st.write("Top 10 RL_at_stop (falladas) para validar X final:")
    # if not failed_with_stop.empty:
    #     st.dataframe(failed_with_stop.sort_values("RL_at_stop", ascending=False)[["Well_ID","Run_Date","Stop_Date","RL_at_stop"]].head(10))
    # else:
    #     st.write("No hay registros con Stop_Date para mostrar.")
    
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
