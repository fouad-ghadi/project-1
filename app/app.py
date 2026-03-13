import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PATHS  (works regardless of where you run from)
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nouvelle_dataset_equilibree.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HeartGuard — Insuffisance Cardiaque",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp {
    background: #060b14 !important;
    font-family: 'DM Mono', monospace;
    color: #c8d6e5;
}

/* ── Background: radial glows ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background:
        radial-gradient(ellipse 80% 60% at 10% 40%, rgba(180,20,20,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 70% 50% at 90% 60%, rgba(0,80,200,0.14) 0%, transparent 60%),
        url('https://images.unsplash.com/photo-1530026405186-ed1f139313f8?w=1600&q=60') center/cover no-repeat fixed;
    opacity: 0.55;
}
.stApp::after {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background: rgba(6,11,20,0.82);
}

/* ── Keep Streamlit content above overlays ── */
.main .block-container { position: relative; z-index: 1; max-width: 1180px; padding: 1.5rem 2rem 4rem; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1020; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Expander (input panel) ── */
[data-testid="stExpander"] {
    background: rgba(10,20,40,0.75) !important;
    border: 1px solid rgba(0,180,150,0.25) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #00d4aa !important;
    letter-spacing: 0.08em;
    padding: 0.8rem 1.2rem !important;
}
[data-testid="stExpander"] summary:hover { color: #fff !important; }

/* ── Inputs ── */
.stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
    color: #e8f0fe !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: rgba(0,212,170,0.5) !important;
    box-shadow: 0 0 0 2px rgba(0,212,170,0.12) !important;
}
label { color: #8fa8c8 !important; font-size: 0.78rem !important; letter-spacing: 0.06em; }

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4aa 0%, #0080ff 100%) !important;
    color: #000 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.1em;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.75rem 2.5rem !important;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 0 30px rgba(0,212,170,0.3);
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

/* ── Metric cards ── */
.metric-card {
    background: rgba(10,22,45,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    backdrop-filter: blur(10px);
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(0,212,170,0.3); }
.metric-val  { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#e8f4fd; }
.metric-lbl  { font-size:0.72rem; color:#607d9a; letter-spacing:0.08em; margin-top:0.25rem; }
.metric-flag { font-size:0.75rem; margin-top:0.3rem; }
.flag-ok   { color:#00d4aa; }
.flag-warn { color:#f0a500; }
.flag-crit { color:#e84040; }

/* ── Result cards ── */
.result-card {
    background: rgba(10,22,45,0.82);
    border-radius: 16px;
    padding: 2rem;
    backdrop-filter: blur(14px);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}
.risk-high { border-color: rgba(232,64,64,0.5) !important; box-shadow: 0 0 40px rgba(232,64,64,0.15); }
.risk-low  { border-color: rgba(0,212,170,0.5) !important; box-shadow: 0 0 40px rgba(0,212,170,0.12); }
.risk-label-high { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#e84040; }
.risk-label-low  { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; color:#00d4aa; }
.risk-pct { font-size:3.5rem; font-weight:700; font-family:'Syne',sans-serif; }
.risk-sub { font-size:0.8rem; color:#607d9a; margin-top:0.4rem; letter-spacing:0.06em; }

/* ── SHAP bar chart area ── */
.shap-section {
    background: rgba(10,22,45,0.72);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    backdrop-filter: blur(10px);
    margin-top: 1rem;
}
.shap-title { font-family:'Syne',sans-serif; font-size:0.9rem; font-weight:600; color:#8fa8c8; letter-spacing:0.1em; margin-bottom:1rem; }

/* ── Section headers ── */
.section-title {
    font-family:'Syne',sans-serif; font-weight:700; font-size:0.72rem;
    color:#607d9a; letter-spacing:0.15em; text-transform:uppercase;
    margin: 1.5rem 0 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-title::after { content:''; flex:1; height:1px; background:rgba(255,255,255,0.07); }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }

/* ── Plot backgrounds ── */
[data-testid="stpyplot"] { background: transparent !important; }
[data-testid="stpyplot"] > div { background: transparent !important; }

/* ── Footer ── */
.footer {
    position: fixed; bottom: 0; left: 0; right: 0;
    background: rgba(6,11,20,0.92);
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 0.55rem 2rem;
    display: flex; justify-content: space-between; align-items: center;
    z-index: 100; backdrop-filter: blur(10px);
    font-size: 0.72rem; color: #3a5070; letter-spacing: 0.05em;
}
.footer span { color: #607d9a; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA & TRAIN / LOAD MODEL
# ─────────────────────────────────────────────
FEATURES = [
    "age", "anaemia", "creatinine_phosphokinase", "diabetes",
    "ejection_fraction", "high_blood_pressure", "platelets",
    "serum_creatinine", "serum_sodium", "sex", "smoking", "time"
]
TARGET = "DEATH_EVENT"

@st.cache_resource(show_spinner=False)
def load_model_and_data():
    """Load or train model. Returns (model, df, demo_mode)."""
    demo = False
    df   = None

    # ── Load dataset ──
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # Synthetic fallback
        demo = True
        rng = np.random.default_rng(42)
        n   = 406
        df  = pd.DataFrame({
            "age":                        rng.integers(40, 95, n).astype(float),
            "anaemia":                    rng.integers(0, 2, n),
            "creatinine_phosphokinase":   rng.integers(23, 7861, n),
            "diabetes":                   rng.integers(0, 2, n),
            "ejection_fraction":          rng.integers(14, 80, n),
            "high_blood_pressure":        rng.integers(0, 2, n),
            "platelets":                  rng.uniform(25100, 850000, n),
            "serum_creatinine":           rng.uniform(0.5, 9.4, n),
            "serum_sodium":               rng.integers(113, 148, n).astype(float),
            "sex":                        rng.integers(0, 2, n),
            "smoking":                    rng.integers(0, 2, n),
            "time":                       rng.integers(4, 285, n),
            "DEATH_EVENT":                rng.integers(0, 2, n),
        })

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Try loading saved model ──
    model = None
    try:
        import joblib
        model = joblib.load(MODEL_PATH)
    except Exception:
        pass

    if model is None:
        model = RandomForestClassifier(n_estimators=200, max_depth=8,
                                       class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        try:
            import joblib
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump(model, MODEL_PATH)
        except Exception:
            pass

    acc  = accuracy_score(y_test, model.predict(X_test))
    surv = int((y == 0).sum())
    dead = int((y == 1).sum())
    return model, df, acc, surv, dead, demo


model, df, accuracy, n_surv, n_dead, demo_mode = load_model_and_data()
n_total = n_surv + n_dead


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
if demo_mode:
    st.warning("⚠️ Dataset non trouvé — mode démo actif avec données synthétiques. "
               "Placez `nouvelle_dataset_equilibree.csv` dans le dossier `data/`.")

col_logo, col_title, col_acc = st.columns([0.08, 0.72, 0.20])
with col_logo:
    st.markdown("""
    <div style="width:52px;height:52px;background:linear-gradient(135deg,#c0392b,#8e1a1a);
         border-radius:14px;display:flex;align-items:center;justify-content:center;
         box-shadow:0 0 20px rgba(192,57,43,0.4);margin-top:4px;">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path d="M12 21C12 21 3 14.5 3 8.5C3 5.42 5.42 3 8.5 3C10.24 3 11.8 3.8 12 5
                 C12.2 3.8 13.76 3 15.5 3C18.58 3 21 5.42 21 8.5C21 14.5 12 21 12 21Z"
              fill="white" opacity="0.95"/>
        <polyline points="3,8.5 7,8.5 9,5 11,11 13,7 15,9.5 21,8.5"
                  stroke="rgba(192,57,43,0.8)" stroke-width="1.2" fill="none"
                  stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>""", unsafe_allow_html=True)

with col_title:
    st.markdown(f"""
    <div style="padding-top:2px;">
      <div style="font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:800;
           color:#f0f6ff;letter-spacing:-0.01em;line-height:1.15;">
        HeartGuard <span style="color:#e84040;">—</span> Insuffisance Cardiaque
      </div>
      <div style="font-size:0.76rem;color:#607d9a;margin-top:3px;letter-spacing:0.04em;">
        Random Forest &nbsp;·&nbsp; {n_total} patients &nbsp;·&nbsp;
        Dataset {'équilibré' if abs(n_surv-n_dead)<20 else 'déséquilibré'}
        &nbsp;·&nbsp; {n_surv} survivants / {n_dead} décédés
        &nbsp;·&nbsp; <span style="color:#00d4aa;">● SYSTÈME ACTIF</span>
      </div>
    </div>""", unsafe_allow_html=True)

with col_acc:
    st.markdown(f"""
    <div style="text-align:right;padding-top:4px;">
      <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
           color:#00d4aa;line-height:1;">{accuracy*100:.1f}%</div>
      <div style="font-size:0.7rem;color:#607d9a;letter-spacing:0.06em;">PRÉCISION DU MODÈLE</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INPUT PANEL  (expander)
# ─────────────────────────────────────────────
with st.expander("≡  Saisie des données patient — cliquer pour ouvrir / fermer", expanded=True):
    r1 = st.columns(6)
    r2 = st.columns(6)

    with r1[0]: age    = st.number_input("Âge", 18, 100, 60)
    with r1[1]: ef     = st.number_input("Fraction d'éjection (%)", 10, 80, 38)
    with r1[2]: creat  = st.number_input("Créatinine sérique", 0.5, 15.0, 1.2, step=0.1)
    with r1[3]: sodium = st.number_input("Sodium sérique", 100, 150, 137)
    with r1[4]: cpk    = st.number_input("CPK (U/L)", 10, 10000, 250)
    with r1[5]: plat   = st.number_input("Plaquettes (×10³)", 50.0, 900.0, 265.0, step=5.0)

    with r2[0]: time_f = st.number_input("Période suivi (jours)", 1, 300, 130)
    with r2[1]: sex    = st.selectbox("Sexe", ["Homme (1)", "Femme (0)"])
    with r2[2]: anaem  = st.selectbox("Anémie", ["Non (0)", "Oui (1)"])
    with r2[3]: diab   = st.selectbox("Diabète", ["Non (0)", "Oui (1)"])
    with r2[4]: hbp    = st.selectbox("Hypertension", ["Non (0)", "Oui (1)"])
    with r2[5]: smoke  = st.selectbox("Tabagisme", ["Non (0)", "Oui (1)"])

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_btn = st.button("⚡  Lancer la prédiction")


# ─────────────────────────────────────────────
#  HELPER: build input vector
# ─────────────────────────────────────────────
def build_input():
    return pd.DataFrame([{
        "age":                      age,
        "anaemia":                  1 if "Oui" in anaem else 0,
        "creatinine_phosphokinase": cpk,
        "diabetes":                 1 if "Oui" in diab else 0,
        "ejection_fraction":        ef,
        "high_blood_pressure":      1 if "Oui" in hbp else 0,
        "platelets":                plat * 1000,
        "serum_creatinine":         creat,
        "serum_sodium":             sodium,
        "sex":                      1 if "Homme" in sex else 0,
        "smoking":                  1 if "Oui" in smoke else 0,
        "time":                     time_f,
    }])


# ─────────────────────────────────────────────
#  PREDICTION  +  DASHBOARD
# ─────────────────────────────────────────────
if predict_btn:
    patient = build_input()
    prob    = float(model.predict_proba(patient)[0][1])
    pred    = int(model.predict(patient)[0])
    high    = pred == 1

    left, right = st.columns([1, 1.4], gap="large")

    # ── Left: result card ──
    with left:
        st.markdown(f"""
        <div class="result-card {'risk-high' if high else 'risk-low'}">
          <div style="font-size:0.72rem;color:#607d9a;letter-spacing:0.12em;margin-bottom:0.5rem;">
            RÉSULTAT DE PRÉDICTION
          </div>
          <div class="{'risk-label-high' if high else 'risk-label-low'}">
            {'🔴 RISQUE ÉLEVÉ' if high else '🟢 RISQUE FAIBLE'}
          </div>
          <div class="risk-pct" style="color:{'#e84040' if high else '#00d4aa'}">
            {prob*100:.1f}%
          </div>
          <div class="risk-sub">probabilité de décès</div>
          <div style="margin-top:1.2rem;font-size:0.8rem;color:#8fa8c8;line-height:1.6;">
            {'⚠️ Ce patient présente un risque significatif. Une évaluation clinique approfondie est recommandée.' 
             if high else 
             '✅ Risque faible détecté. Continuer le suivi de routine.'}
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Vital flags ──
        st.markdown('<div class="section-title">Indicateurs cliniques</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        flags = [
            ("Fraction éjection", ef,    "<30", ef < 30,   "🔴 Critique (<30%)" if ef<30 else ("🟡 Attention" if ef<45 else "🟢 Normal"), ef<30),
            ("Créatinine",        creat, ">1.5", creat>1.5, "🔴 Élevée" if creat>1.5 else "🟢 Normale",   creat>1.5),
            ("Sodium sérique",    sodium,"<135", sodium<135,"🔴 Hyponatrémie" if sodium<135 else "🟢 Normal", sodium<135),
            ("CPK",               cpk,  ">500", cpk>500,   "🔴 Élevée" if cpk>500 else "🟢 Normale",     cpk>500),
        ]
        for i, (lbl, val, thr, alert, msg, _) in enumerate(flags):
            col = c1 if i % 2 == 0 else c2
            with col:
                cls = "flag-crit" if alert else "flag-ok"
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:0.6rem;">
                  <div class="metric-val">{val}</div>
                  <div class="metric-lbl">{lbl}</div>
                  <div class="metric-flag {cls}">{msg}</div>
                </div>""", unsafe_allow_html=True)

    # ── Right: SHAP-style importance ──
    with right:
        st.markdown('<div class="shap-section">', unsafe_allow_html=True)
        st.markdown('<div class="shap-title">📊 IMPACT DES VARIABLES — PATIENT ACTUEL</div>',
                    unsafe_allow_html=True)

        try:
            import shap, matplotlib.pyplot as plt, matplotlib as mpl
            mpl.rcParams.update({
                "figure.facecolor": "none", "axes.facecolor": "none",
                "savefig.facecolor":"none",  "text.color": "#c8d6e5",
                "axes.labelcolor":  "#c8d6e5", "xtick.color": "#8fa8c8",
                "ytick.color":      "#c8d6e5", "axes.edgecolor": "none",
                "font.family":      "monospace",
            })
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(patient)
            vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

            feat_labels = [
                "Âge","Anémie","CPK","Diabète","Fr. Éjection",
                "Hypertension","Plaquettes","Créatinine","Sodium","Sexe","Tabac","Suivi"
            ]
            pairs  = sorted(zip(np.abs(vals), vals, feat_labels), reverse=True)[:8]
            labels = [p[2] for p in pairs]
            colors = ["#e84040" if p[1] > 0 else "#00d4aa" for p in pairs]
            absv   = [p[0] for p in pairs]

            fig, ax = plt.subplots(figsize=(6, 3.4))
            bars = ax.barh(labels[::-1], absv[::-1], color=colors[::-1],
                           height=0.55, edgecolor="none")
            for bar, v in zip(bars, absv[::-1]):
                ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
                        f"{v:.4f}", va="center", ha="left", fontsize=8, color="#8fa8c8")
            ax.set_xlim(0, max(absv) * 1.28)
            ax.tick_params(labelsize=9)
            ax.xaxis.set_visible(False)
            ax.grid(False)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        except ImportError:
            # Fallback: feature importance bars
            import matplotlib.pyplot as plt, matplotlib as mpl
            mpl.rcParams.update({
                "figure.facecolor":"none","axes.facecolor":"none",
                "savefig.facecolor":"none","text.color":"#c8d6e5",
                "axes.labelcolor":"#c8d6e5","xtick.color":"#8fa8c8",
                "ytick.color":"#c8d6e5","axes.edgecolor":"none","font.family":"monospace",
            })
            imp    = model.feature_importances_
            labels = ["Âge","Anémie","CPK","Diabète","Fr. Éjection",
                      "Hypertension","Plaquettes","Créatinine","Sodium","Sexe","Tabac","Suivi"]
            pairs  = sorted(zip(imp, labels), reverse=True)[:8]
            vals2  = [p[0] for p in pairs]
            lbl2   = [p[1] for p in pairs]
            colors = ["#e84040" if v > np.median(imp) else "#00d4aa" for v in vals2]

            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.barh(lbl2[::-1], vals2[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.tick_params(labelsize=9)
            ax.xaxis.set_visible(False)
            ax.grid(False)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Top 3 driving factors ──
        st.markdown('<div class="section-title">Facteurs principaux</div>', unsafe_allow_html=True)
        feat_labels_full = ["Âge","Anémie","CPK","Diabète","Fr. Éjection",
                            "Hypertension","Plaquettes","Créatinine","Sodium","Sexe","Tabac","Suivi"]
        top3_idx  = np.argsort(model.feature_importances_)[::-1][:3]
        top3_vals = [feat_labels_full[i] for i in top3_idx]
        c1, c2, c3 = st.columns(3)
        for col, name, rank in zip([c1, c2, c3], top3_vals, ["#e84040","#f0a500","#00d4aa"]):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border-color:rgba(255,255,255,0.1)">
                  <div style="font-size:1.5rem;font-weight:700;color:{rank};font-family:'Syne',sans-serif;">
                    #{top3_vals.index(name)+1}
                  </div>
                  <div class="metric-lbl" style="color:#c8d6e5;">{name}</div>
                </div>""", unsafe_allow_html=True)

else:
    # ── Idle state ──
    st.markdown("""
    <div style="text-align:center;padding:3.5rem 0 2rem;">
      <svg width="56" height="56" viewBox="0 0 24 24" fill="none" style="opacity:0.25;margin-bottom:1rem;">
        <path d="M12 21C12 21 3 14.5 3 8.5C3 5.42 5.42 3 8.5 3C10.24 3 11.8 3.8 12 5
                 C12.2 3.8 13.76 3 15.5 3C18.58 3 21 5.42 21 8.5C21 14.5 12 21 12 21Z"
              fill="#607d9a"/>
      </svg>
      <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:600;color:#3a5070;">
        Saisissez les données du patient et lancez la prédiction
      </div>
      <div style="font-size:0.8rem;color:#2a3a50;margin-top:0.5rem;">
        Le modèle analysera 12 variables cliniques et retournera le risque avec explications SHAP
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MODEL PERFORMANCE TABLE  (expander)
# ─────────────────────────────────────────────
with st.expander("📈  Performances du modèle — voir les métriques"):
    st.markdown("""
    | Modèle | Accuracy | ROC-AUC | F1-Score | Note |
    |--------|----------|---------|----------|------|
    | **Random Forest** ✅ | **~85%** | **~0.91** | **~0.84** | Sélectionné — meilleur compromis |
    | XGBoost | ~83% | ~0.90 | ~0.82 | Très proche |
    | LightGBM | ~82% | ~0.89 | ~0.81 | Rapide, léger |
    | Régression Logistique | ~78% | ~0.85 | ~0.76 | Baseline |
    """)
    st.caption("Évaluation sur 20% du dataset (80 patients). Dataset équilibré via SMOTE si nécessaire.")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>HeartGuard &nbsp;·&nbsp; Clinical Decision Support System</span>
  <span>
    Fouad Ghadi &nbsp;·&nbsp; Yassine Ait Bella &nbsp;·&nbsp; Rabi Ilyas &nbsp;·&nbsp;
    Yahiaoui Ziyad &nbsp;·&nbsp; Chakir Mohamed
  </span>
  <span>Centrale Casablanca &nbsp;·&nbsp; Coding Week &nbsp;·&nbsp; Mars 2026</span>
</div>
""", unsafe_allow_html=True)
