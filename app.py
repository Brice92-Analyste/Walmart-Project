"""
app.py — Walmart Sales Forecast
Streamlit application for weekly sales prediction.

Usage:
    streamlit run app.py

Requirements:
    pip install streamlit pandas numpy xgboost scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import random
from datetime import timedelta

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="🛒",
    layout="wide",
)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ARTIFACTS_DIR  = "./model_artifacts/"
HISTORY_MONTHS = 2          # Mois d'historique conservés pour les lags
N_DEPTS        = 6          # Départements à prédire
HOLDOUT_DATE   = pd.Timestamp("2012-10-26")   # Semaine jamais vue par le modèle

FEATURES = [
    "Store", "Dept", "Size", "Type",
    "IsHoliday", "Is_SuperBowl", "Is_LaborDay", "Is_Chrismas", "Is_Thankgiving",
    "Temperature", "Fuel_Price", "CPI", "Unemployment",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Promotion_Active", "Nombre_Promotion", "Is_Vente",
    "Years", "Week_sin", "Week_cos", "Month_sin", "Month_cos",
    "Lag_1", "Lag_2", "Lag_4",
    "Rolling_mean_4", "Rolling_mean_8",
    "Rolling_std_4",  "Rolling_std_8", "Store_Dept_Encoded"
]

# ─────────────────────────────────────────────
# LOADERS (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement du modèle…")
def load_artifacts():
    model    = pickle.load(open(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"),          "rb"))
    features = pickle.load(open(os.path.join(ARTIFACTS_DIR, "features.pkl"),           "rb"))
    le_type  = pickle.load(open(os.path.join(ARTIFACTS_DIR, "label_encoder_type.pkl"), "rb"))
    te_dict  = pickle.load(open(os.path.join(ARTIFACTS_DIR, "target_encoding.pkl"),    "rb"))
    holidays = json.load(open(os.path.join(ARTIFACTS_DIR, "holidays.json")))
    return model, features, le_type, te_dict, holidays


@st.cache_data(show_spinner="Chargement de l'historique des ventes…")
def load_history():
    """
    Charge l'historique et conserve les 2 derniers mois.
    Ces données servent UNIQUEMENT au calcul des lags — jamais à l'entraînement.
    """
    hist = pd.read_json(os.path.join(ARTIFACTS_DIR, "historique_ventes.json"))
    hist["Date"] = pd.to_datetime(hist["Date"])
    hist = hist.sort_values(["Store", "Dept", "Date"]).reset_index(drop=True)
    cutoff = hist["Date"].max() - pd.DateOffset(months=HISTORY_MONTHS)
    return hist[hist["Date"] >= cutoff].copy()


@st.cache_data(show_spinner="Chargement du fichier test…")
def load_test_data():
    test    = pd.read_csv("data/test.csv")
    stores  = pd.read_csv("data/stores.csv")
    feature = pd.read_csv("data/features.csv")

    df_test = test.merge(stores, on="Store", how="left")
    df_test = df_test.merge(feature, on=["Store", "Date"], how="left")

    if "IsHoliday_x" in df_test.columns:
        df_test["IsHoliday"] = df_test["IsHoliday_x"] | df_test["IsHoliday_y"]
        df_test = df_test.drop(columns=["IsHoliday_x", "IsHoliday_y"])

    df_test["Date"] = pd.to_datetime(df_test["Date"])
    return df_test


@st.cache_data(show_spinner="Chargement du hold-out…")
def load_holdout():
    """
    Charge la semaine réelle 2012-10-26 (jamais vue par le modèle).
    Retourne None si le fichier n'existe pas encore.
    """
    path = os.path.join(ARTIFACTS_DIR, "holdout_semaine.json")
    if not os.path.exists(path):
        return None
    ho = pd.read_json(path)
    ho["Date"] = pd.to_datetime(ho["Date"])
    return ho


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def flag_holidays(date: pd.Timestamp, holidays: dict) -> dict:
    d = str(date.date())
    return {
        "Is_SuperBowl":   int(d in holidays["superbowl"]),
        "Is_LaborDay":    int(d in holidays["laborday"]),
        "Is_Chrismas":    int(d in holidays["noel"]),
        "Is_Thankgiving": int(d in holidays["thanksgiving"]),
        "IsHoliday":      int(
            d in holidays["superbowl"] or d in holidays["laborday"] or
            d in holidays["noel"]      or d in holidays["thanksgiving"]
        ),
    }


def compute_lag_features(history_store_dept: pd.DataFrame,
                          target_date: pd.Timestamp) -> dict:
    hist  = history_store_dept[history_store_dept["Date"] < target_date].sort_values("Date")
    sales = hist["Weekly_Sales"].values

    def _get(offset):
        idx = len(sales) - offset
        return float(sales[idx]) if idx >= 0 else (float(np.nanmean(sales)) if len(sales) > 0 else 0.0)

    lag1, lag2, lag4 = _get(1), _get(2), _get(4)
    w4 = sales[-4:] if len(sales) >= 4 else sales
    w8 = sales[-8:] if len(sales) >= 8 else sales

    return {
        "Lag_1":          lag1,
        "Lag_2":          lag2,
        "Lag_4":          lag4,
        "Rolling_mean_4": float(np.mean(w4)) if len(w4) > 0 else lag1,
        "Rolling_mean_8": float(np.mean(w8)) if len(w8) > 0 else lag1,
        "Rolling_std_4":  float(np.std(w4))  if len(w4) > 1 else 0.0,
        "Rolling_std_8":  float(np.std(w8))  if len(w8) > 1 else 0.0,
    }


def build_row(store_id, dept_id, pred_date, df_test_store,
              history, holidays, te_dict, le_type):
    ctx = df_test_store[df_test_store["Date"] == pred_date]
    if ctx.empty:
        ctx = df_test_store.sort_values("Date").iloc[[-1]]
    ctx = ctx.iloc[0]

    md_vals = {
        f"MarkDown{i}": (
            float(ctx.get(f"MarkDown{i}", 0))
            if not pd.isna(ctx.get(f"MarkDown{i}", np.nan)) else 0.0
        ) for i in range(1, 6)
    }

    row = {
        "Store":        store_id,
        "Dept":         dept_id,
        "Size":         ctx.get("Size", 0),
        "Type":         le_type.transform([ctx["Type"]])[0] if "Type" in ctx.index else 0,
        "Temperature":  ctx.get("Temperature",  0),
        "Fuel_Price":   ctx.get("Fuel_Price",   0),
        "CPI":          ctx.get("CPI",          0),
        "Unemployment": ctx.get("Unemployment", 0),
        **md_vals,
    }

    row["Promotion_Active"] = int(any(row[f"MarkDown{i}"] > 0 for i in range(1, 6)))
    row["Nombre_Promotion"] = sum(1 for i in range(1, 6) if row[f"MarkDown{i}"] > 0)
    row["Is_Vente"]         = 1

    row.update(flag_holidays(pred_date, holidays))

    week = pred_date.isocalendar().week
    mois = pred_date.month
    row["Years"]     = pred_date.year
    row["Week_sin"]  = np.sin(2 * np.pi * week / 52)
    row["Week_cos"]  = np.cos(2 * np.pi * week / 52)
    row["Month_sin"] = np.sin(2 * np.pi * mois / 12)
    row["Month_cos"] = np.cos(2 * np.pi * mois / 12)

    hist_sd = history[(history["Store"] == store_id) & (history["Dept"] == dept_id)]
    row.update(compute_lag_features(hist_sd, pred_date))

    row["Store_Dept_Encoded"] = te_dict["map"].get((store_id, dept_id), te_dict["global_mean"])
    return row


# ─────────────────────────────────────────────
# PRÉDICTION
# ─────────────────────────────────────────────

def predict_week(store_id, dept_ids, pred_date,
                 df_test, history, holidays, te_dict, le_type, model, features):
    df_store = df_test[df_test["Store"] == store_id].copy()
    rows = [
        build_row(store_id, dept, pred_date, df_store, history, holidays, te_dict, le_type)
        for dept in dept_ids
    ]
    X_pred    = pd.DataFrame(rows)[features]
    log_preds = model.predict(X_pred)
    preds     = np.expm1(log_preds)

    return pd.DataFrame({
        "Store":               store_id,
        "Département":         dept_ids,
        "Semaine":             pred_date.strftime("%Y-%m-%d"),
        "Ventes Prédites ($)": np.round(preds, 2),
        "Lag_1 ($)":           [r["Lag_1"]          for r in rows],
        "Rolling_mean_4 ($)":  [r["Rolling_mean_4"] for r in rows],
    })


# ─────────────────────────────────────────────
# MÉTRIQUES DE COMPARAISON
# ─────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    r2   = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape, "R²": r2}


# ─────────────────────────────────────────────
# UI — HEADER
# ─────────────────────────────────────────────

st.title("🛒 Walmart — Prévision des Ventes Hebdomadaires")
st.markdown(
    "Application de prédiction basée sur un modèle **XGBoost** entraîné "
    "sur les données historiques Walmart (2010–2012)."
)
st.divider()

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────

try:
    model, features, le_type, te_dict, holidays = load_artifacts()
    history  = load_history()
    df_test  = load_test_data()
    holdout  = load_holdout()
except FileNotFoundError as e:
    st.error(
        f" Fichier manquant : `{e.filename}`\n\n"
        "Vérifiez que `./model_artifacts/` et `./data/` contiennent tous les fichiers nécessaires."
    )
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — PARAMÈTRES
# ─────────────────────────────────────────────

with st.sidebar:
    st.header(" Paramètres")
    st.divider()

    # ── Mode ──────────────────────────────────
    mode = st.radio(
        "Mode",
        [" Prédiction libre", "🎯 Évaluation Hold-out (2012-10-26)"],
        help=(
            "**Prédiction libre** : choisissez librement boutique, départements et semaine.\n\n"
            "**Évaluation Hold-out** : teste le modèle sur la dernière semaine réelle "
            "(jamais vue lors de l'entraînement) et compare prédit vs réel."
        )
    )
    is_holdout_mode = mode.startswith("🎯")

    st.divider()

    # ── Boutique ──────────────────────────────
    all_stores = sorted(df_test["Store"].unique().tolist())

    if is_holdout_mode:
        st.markdown("**🏪 Boutique**")
        mode_store = st.radio("Sélection", ["Aléatoire", "Manuelle"], horizontal=True, key="ms_ho")
        if mode_store == "Aléatoire":
            if st.button(" Tirer au sort", use_container_width=True, key="btn_store_ho"):
                st.session_state["ho_store"] = random.choice(all_stores)
            selected_store = st.session_state.get("ho_store", random.choice(all_stores))
        else:
            selected_store = st.selectbox("Boutique", all_stores, key="sel_store_ho")
        st.info(f"🏪 **Boutique #{selected_store}** | 📅 **2012-10-26** (hold-out)")
        pred_date = HOLDOUT_DATE
    else:
        mode_store = st.radio("Sélection de la boutique", ["Aléatoire", "Manuelle"], horizontal=True)
        if mode_store == "Aléatoire":
            if st.button(" Tirer une boutique au sort", use_container_width=True):
                st.session_state["selected_store"] = random.choice(all_stores)
            selected_store = st.session_state.get("selected_store", random.choice(all_stores))
        else:
            selected_store = st.selectbox("Choisir la boutique", all_stores)
        st.info(f"🏪 **Boutique sélectionnée : #{selected_store}**")

    # ── Départements ──────────────────────────
    st.divider()
    all_depts = sorted(df_test[df_test["Store"] == selected_store]["Dept"].unique().tolist())

    if is_holdout_mode and holdout is not None:
        ho_depts = sorted(holdout[holdout["Store"] == selected_store]["Dept"].unique().tolist())
        available_depts = ho_depts if ho_depts else all_depts
    else:
        available_depts = all_depts

    mode_dept = st.radio(
        "Sélection des départements",
        ["Aléatoire (6)", "Manuelle"],
        horizontal=True
    )

    key_depts = "ho_depts" if is_holdout_mode else "selected_depts"
    if mode_dept == "Aléatoire (6)":
        if st.button(" Tirer 6 départements", use_container_width=True):
            n = min(N_DEPTS, len(available_depts))
            st.session_state[key_depts] = random.sample(available_depts, n)
        selected_depts = st.session_state.get(
            key_depts,
            random.sample(available_depts, min(N_DEPTS, len(available_depts)))
        )
    else:
        selected_depts = st.multiselect(
            "Choisir les départements (max 6)",
            available_depts,
            default=available_depts[:min(6, len(available_depts))],
            max_selections=6
        )

    if selected_depts:
        st.success(f" Départements : {selected_depts}")

    # ── Semaine (mode libre uniquement) ───────
    if not is_holdout_mode:
        st.divider()
        last_test_date  = df_test["Date"].max()
        pred_date_input = st.date_input(
            "📅 Semaine à prédire",
            value=last_test_date,
            min_value=df_test["Date"].min().date(),
            max_value=(last_test_date + timedelta(weeks=4)).date(),
        )
        pred_date = pd.Timestamp(pred_date_input)

    # ── Bouton ────────────────────────────────
    st.divider()
    btn_label  = " Comparer Prédit vs Réel" if is_holdout_mode else "🚀 Lancer la Prédiction"
    run_button = st.button(btn_label, use_container_width=True, type="primary")

    # ── Avertissement si hold-out absent ──────
    if is_holdout_mode and holdout is None:
        st.warning(
            " `holdout_semaine.json` introuvable.\n\n"
            "Ajoutez ce bloc dans votre notebook avant la sauvegarde :\n"
            "```python\n"
            "holdout_save = holdout[\n"
            "  ['Store','Dept','Date','Weekly_Sales']\n"
            "].copy()\n"
            "holdout_save['Date'] = holdout_save['Date'].astype(str)\n"
            "holdout_save.to_json(\n"
            "  './model_artifacts/holdout_semaine.json',\n"
            "  orient='records', indent=2)\n"
            "```"
        )

# ─────────────────────────────────────────────
# MAIN — RÉSULTATS
# ─────────────────────────────────────────────

if run_button:
    if not selected_depts:
        st.warning(" Veuillez sélectionner au moins un département.")
        st.stop()

    with st.spinner("Calcul des prédictions en cours…"):
        results = predict_week(
            selected_store, selected_depts, pred_date,
            df_test, history, holidays, te_dict, le_type, model, features
        )

    # ── Titre ─────────────────────────────────────────────────────────────
    badge = " Mode Évaluation Hold-out" if is_holdout_mode else " Mode Prédiction"
    st.subheader(
        f"{badge} — Boutique #{selected_store} | Semaine du {pred_date.strftime('%d %b %Y')}"
    )

    # ─────────────────────────────────────────────────────────────────────
    # MODE HOLD-OUT : comparaison prédit vs réel
    # ─────────────────────────────────────────────────────────────────────
    if is_holdout_mode and holdout is not None:

        ho_store = holdout[
            (holdout["Store"] == selected_store) &
            (holdout["Dept"].isin(selected_depts))
        ][["Dept", "Weekly_Sales"]].rename(
            columns={"Dept": "Département", "Weekly_Sales": "Ventes Réelles ($)"}
        )

        # Fusion prédit + réel
        compare = results.merge(ho_store, on="Département", how="left")
        compare["Erreur ($)"] = (compare["Ventes Prédites ($)"] - compare["Ventes Réelles ($)"]).round(2)
        compare["Erreur (%)"] = (
            (compare["Erreur ($)"] / compare["Ventes Réelles ($)"].replace(0, np.nan)) * 100
        ).round(2)
        compare["Statut"] = compare["Erreur (%)"].apply(
            lambda e: "✅ Bon" if abs(e) <= 10 else ("⚠️ Acceptable" if abs(e) <= 20 else "❌ Écart élevé")
            if pd.notna(e) else "N/A"
        )

        # ── KPIs ──────────────────────────────────────────────────────────
        total_pred = compare["Ventes Prédites ($)"].sum()
        total_real = compare["Ventes Réelles ($)"].sum()
        has_real   = compare["Ventes Réelles ($)"].notna().any()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(" Total Prédit", f"${total_pred:,.0f}")
        if has_real:
            delta_pct = ((total_pred - total_real) / total_real * 100) if total_real else 0
            col2.metric("✅ Total Réel",   f"${total_real:,.0f}")
            col3.metric("📏 Écart Total",  f"${total_pred - total_real:,.0f}",
                        delta=f"{delta_pct:+.1f}%", delta_color="inverse")
        col4.metric(" Départements", len(selected_depts))

        st.divider()

        # ── Tableau de comparaison ────────────────────────────────────────
        col_left, col_right = st.columns([1.4, 1])

        with col_left:
            st.markdown("###  Comparaison Département par Département")
            disp = compare[[
                "Département", "Ventes Prédites ($)", "Ventes Réelles ($)",
                "Erreur ($)", "Erreur (%)", "Statut"
            ]].copy()
            for c in ["Ventes Prédites ($)", "Ventes Réelles ($)", "Erreur ($)"]:
                disp[c] = disp[c].apply(lambda v: f"${v:,.2f}" if pd.notna(v) else "N/A")
            disp["Erreur (%)"] = disp["Erreur (%)"].apply(
                lambda v: f"{v:+.1f}%" if pd.notna(v) else "N/A"
            )
            st.dataframe(disp, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("###  Prédit vs Réel par Département")
            valid_chart = compare.dropna(subset=["Ventes Réelles ($)"])
            if not valid_chart.empty:
                chart_df = valid_chart.set_index("Département")[
                    ["Ventes Prédites ($)", "Ventes Réelles ($)"]
                ]
                st.bar_chart(chart_df, use_container_width=True)
            else:
                st.info("Aucune valeur réelle disponible pour les départements sélectionnés.")

        st.divider()

        # ── Métriques de performance ──────────────────────────────────────
        valid = compare.dropna(subset=["Ventes Réelles ($)"])
        if len(valid) > 0:
            st.markdown("###  Métriques de Performance (Hold-out)")
            m = compute_metrics(
                valid["Ventes Réelles ($)"].values,
                valid["Ventes Prédites ($)"].values
            )
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("MAE",  f"${m['MAE']:,.0f}",     help="Erreur absolue moyenne")
            mc2.metric("RMSE", f"${m['RMSE']:,.0f}",    help="Racine de l'erreur quadratique moyenne")
            mc3.metric("MAPE", f"{m['MAPE (%)']:.1f}%", help="Erreur en pourcentage")
            mc4.metric("R²",   f"{m['R²']:.4f}",        help="Coefficient de détermination")

            mape_val = m["MAPE (%)"]
            if mape_val <= 10:
                st.success(f" Excellente précision — MAPE de {mape_val:.1f}% (≤ 10%)")
            elif mape_val <= 20:
                st.warning(f" Précision acceptable — MAPE de {mape_val:.1f}% (≤ 20%)")
            else:
                st.error(f" Précision insuffisante — MAPE de {mape_val:.1f}% (> 20%)")

    elif is_holdout_mode and holdout is None:
        st.info(
            " Le fichier `holdout_semaine.json` est manquant. "
            "Consultez le message dans la sidebar pour le générer."
        )

    # ─────────────────────────────────────────────────────────────────────
    # MODE LIBRE
    # ─────────────────────────────────────────────────────────────────────
    else:
        total_pred = results["Ventes Prédites ($)"].sum()
        max_dept   = results.loc[results["Ventes Prédites ($)"].idxmax(), "Département"]
        max_vente  = results["Ventes Prédites ($)"].max()
        min_dept   = results.loc[results["Ventes Prédites ($)"].idxmin(), "Département"]
        min_vente  = results["Ventes Prédites ($)"].min()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(" Ventes Totales Prédites",    f"${total_pred:,.0f}")
        col2.metric(" Départements Analysés",      len(selected_depts))
        col3.metric(f" Meilleur Dept #{max_dept}",  f"${max_vente:,.0f}")
        col4.metric(f" Moins Bon Dept #{min_dept}", f"${min_vente:,.0f}")

        st.divider()

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            st.markdown("###  Détail par Département")
            disp = results[["Département", "Ventes Prédites ($)", "Lag_1 ($)", "Rolling_mean_4 ($)"]].copy()
            for c in ["Ventes Prédites ($)", "Lag_1 ($)", "Rolling_mean_4 ($)"]:
                disp[c] = disp[c].map("${:,.2f}".format)
            st.dataframe(disp, use_container_width=True, hide_index=True)

        with col_right:
            st.markdown("###  Ventes Prédites par Département")
            st.bar_chart(
                results.set_index("Département")[["Ventes Prédites ($)"]],
                use_container_width=True
            )

    # ── Historique (commun aux deux modes) ────────────────────────────────
    st.divider()
    st.markdown("###  Historique des 2 Derniers Mois (données utilisées pour les lags)")
    hist_store = history[
        (history["Store"] == selected_store) &
        (history["Dept"].isin(selected_depts))
    ]
    if hist_store.empty:
        st.info("Aucun historique disponible pour ces départements dans la fenêtre des 2 derniers mois.")
    else:
        pivot = (
            hist_store.groupby(["Date", "Dept"])["Weekly_Sales"]
            .sum().unstack("Dept").sort_index()
        )
        pivot.columns = [f"Dept {c}" for c in pivot.columns]
        st.line_chart(pivot, use_container_width=True)

    # ── Fêtes ─────────────────────────────────────────────────────────────
    hflags = flag_holidays(pred_date, holidays)
    fetes  = [k for k, v in hflags.items() if v == 1 and k != "IsHoliday"]
    labels_fetes = {
        "Is_SuperBowl":   " Super Bowl",
        "Is_LaborDay":    " Labor Day",
        "Is_Chrismas":    " Noël",
        "Is_Thankgiving": " Thanksgiving",
    }
    if fetes:
        st.info(f" **Semaine de fête :** {' · '.join(labels_fetes.get(f, f) for f in fetes)}")
    else:
        st.caption(" Semaine ordinaire (aucune fête détectée)")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Modèle XGBoost · Walmart Dataset (Kaggle) · "
    "Historique lags : 2 derniers mois · Hold-out : semaine du 2012-10-26"
)
