import os
import io
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score, log_loss

# =============================================================================
# CONFIG
# =============================================================================
st.set_page_config(page_title="Mini-Kaggle: Adult Income", layout="wide")

# --- Environment switches ---
ADMIN_PASSWORD = os.getenv("LEADERBOARD_ADMIN", "")
DB_PATH = os.getenv("LEADERBOARD_DB", "leaderboard.db")
GROUND_DEFAULT = os.getenv("GROUND_CSV", "ground.csv")
PUBLIC_IDS_DEFAULT = os.getenv("PUBLIC_IDS_CSV", "public_ids.csv")
SUBMISSIONS_DIR = Path(os.getenv("SUBMISSIONS_DIR", "submissions"))
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SUBMISSIONS_PER_TEAM_PER_DAY = int(os.getenv("MAX_SUBS_PER_DAY", "5"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "10")) * 1024 * 1024

COMPETITION_NAME = "Mini-Compétition – Adult Income (ROC AUC / Log Loss)"

# Competition dates (optional - comment out to disable)
COMPETITION_START = datetime.strptime(os.getenv("COMP_START", "2025-01-01"), "%Y-%m-%d")
COMPETITION_END = datetime.strptime(os.getenv("COMP_END", "2025-11-19"), "%Y-%m-%d")

# Thread safety for SQLite
db_lock = Lock()

# =============================================================================
# UTILS
# =============================================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def read_uploaded_csv(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(io.BytesIO(uploaded.read()))

def validate_submission_columns(sub: pd.DataFrame):
    required = {"id", "income_prob"}
    if not required.issubset(set(sub.columns)):
        missing = required - set(sub.columns)
        raise ValueError(f"Colonnes manquantes dans submission: {missing}")

def validate_ground_columns(gt: pd.DataFrame):
    required = {"id", "income"}
    if not required.issubset(set(gt.columns)):
        missing = required - set(gt.columns)
        raise ValueError(f"Colonnes manquantes dans ground: {missing}")

def validate_probs(series: pd.Series):
    if series.isna().any():
        raise ValueError("income_prob contient des NaN.")
    if ((series < 0) | (series > 1)).any():
        raise ValueError("income_prob doit être dans [0,1].")

def align_by_id(gt: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.drop_duplicates(subset=["id"], keep="last")
    gt = gt.drop_duplicates(subset=["id"], keep="last")
    
    # Check for ID mismatches
    expected_ids = set(gt["id"].astype(int))
    submitted_ids = set(sub["id"].astype(int))
    
    missing_ids = expected_ids - submitted_ids
    extra_ids = submitted_ids - expected_ids
    
    if missing_ids:
        raise ValueError(f"❌ {len(missing_ids)} IDs manquants dans votre soumission. "
                        f"Attendu: {len(expected_ids)} IDs, reçu: {len(submitted_ids)} IDs.")
    
    if extra_ids:
        st.warning(f"⚠️ {len(extra_ids)} IDs supplémentaires ignorés (non présents dans le test set).")
    
    df = gt.merge(sub, on="id", how="inner")
    if df.empty:
        raise ValueError("Aucun id commun entre ground et submission.")
    return df

def compute_scores(df):
    y_true = df["income"].astype(int).values
    y_prob = df["income_prob"].astype(float).values
    # clip probs to avoid log(0)
    y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "n": int(len(y_true)),
    }

def split_public_private(df: pd.DataFrame, public_ids: pd.Series | None):
    if public_ids is None:
        return None, None
    pub_ids = set(pd.Series(public_ids).astype(int).tolist())
    pub_df = df[df["id"].astype(int).isin(pub_ids)].copy()
    priv_df = df[~df["id"].astype(int).isin(pub_ids)].copy()
    if pub_df.empty or priv_df.empty:
        return None, None
    return pub_df, priv_df

def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS submissions(
            sid INTEGER PRIMARY KEY AUTOINCREMENT,
            team TEXT,
            filename TEXT,
            timestamp TEXT,
            public_auc REAL,
            private_auc REAL,
            overall_auc REAL,
            logloss REAL,
            n_overall INTEGER,
            n_public INTEGER,
            n_private INTEGER
        )
    """)
    return conn

def insert_submission(conn, row: dict):
    with db_lock:
        conn.execute("""
            INSERT INTO submissions(team, filename, timestamp, public_auc, private_auc, overall_auc, logloss, n_overall, n_public, n_private)
            VALUES(:team, :filename, :timestamp, :public_auc, :private_auc, :overall_auc, :logloss, :n_overall, :n_public, :n_private)
        """, row)
        conn.commit()

def count_team_submissions_today(team: str) -> int:
    conn = get_conn()
    with db_lock:
        df = pd.read_sql_query(
            "SELECT COUNT(*) AS n FROM submissions WHERE LOWER(team)=LOWER(?) AND DATE(timestamp)=DATE('now')",
            conn, params=[team]
        )
    return int(df["n"].iloc[0])

def archive_submission(team: str, uploaded_file) -> Path:
    safe_team = "".join([c for c in team if c.isalnum() or c in (" ", "_", "-", ".")]).strip().replace(" ", "_")
    team_dir = SUBMISSIONS_DIR / safe_team
    team_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    fname = f"{ts}__{getattr(uploaded_file, 'name', 'submission.csv')}"
    dest = team_dir / fname
    uploaded_file.seek(0)
    dest.write_bytes(uploaded_file.read())
    return dest

def check_competition_status():
    """Check if competition is active"""
    now = datetime.now()
    if now < COMPETITION_START:
        return "not_started"
    elif now > COMPETITION_END:
        return "ended"
    return "active"

# =============================================================================
# LOAD HIDDEN FILES (default)
# =============================================================================

ground_df = None
public_ids = None

if os.path.exists(GROUND_DEFAULT):
    try:
        ground_df = load_csv(GROUND_DEFAULT)
    except Exception:
        ground_df = None

if os.path.exists(PUBLIC_IDS_DEFAULT):
    try:
        public_ids = load_csv(PUBLIC_IDS_DEFAULT)["id"]
    except Exception:
        public_ids = None

# =============================================================================
# SIDEBAR – Admin
# =============================================================================
st.sidebar.title("⚙️ Configuration / Admin")

admin_mode = st.sidebar.checkbox("Mode administrateur")
admin_ok = False
if admin_mode:
    pwd = st.sidebar.text_input("Mot de passe admin", type="password")
    admin_ok = (pwd == ADMIN_PASSWORD) if ADMIN_PASSWORD else (pwd != "")

# Admin-only: upload/replace ground & public ids
if admin_ok:
    st.sidebar.success("✅ Admin authentifié")
    
    st.sidebar.subheader("Ground truth (labels cachés)")
    up_ground = st.sidebar.file_uploader("Charger/Remplacer ground.csv (id,income)", type=["csv"], key="ground_admin")
    if up_ground is not None:
        try:
            tmp = read_uploaded_csv(up_ground)
            validate_ground_columns(tmp)
            tmp.to_csv(GROUND_DEFAULT, index=False)
            ground_df = tmp
            st.sidebar.success("✅ ground.csv mis à jour")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Ground invalide: {e}")

    st.sidebar.subheader("Public IDs (optionnel)")
    up_public = st.sidebar.file_uploader("Charger/Remplacer public_ids.csv (id)", type=["csv"], key="public_admin")
    if up_public is not None:
        try:
            tmp = read_uploaded_csv(up_public)
            if "id" not in tmp.columns:
                raise ValueError("Fichier public_ids.csv sans colonne 'id'")
            tmp.to_csv(PUBLIC_IDS_DEFAULT, index=False)
            public_ids = tmp["id"]
            st.sidebar.success("✅ public_ids.csv mis à jour")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Public IDs invalide: {e}")

    st.sidebar.divider()
    
    # Private leaderboard reveal
    reveal_private = st.sidebar.checkbox("🔓 Révéler leaderboard privé", 
                                         help="Affiche les scores sur le test set privé")
    
    # Download team submissions
    st.sidebar.subheader("📥 Télécharger soumissions")
    conn = get_conn()
    with db_lock:
        lb_all = pd.read_sql_query("SELECT DISTINCT team FROM submissions ORDER BY team", conn)
    if not lb_all.empty:
        teams = lb_all['team'].tolist()
        selected_team = st.sidebar.selectbox("Équipe", ["-- Sélectionner --"] + teams)
        if selected_team != "-- Sélectionner --":
            with db_lock:
                team_subs = pd.read_sql_query(
                    "SELECT filename, timestamp FROM submissions WHERE team=? ORDER BY timestamp DESC",
                    conn, params=[selected_team]
                )
            for _, row in team_subs.iterrows():
                file_path = Path(row['filename'])
                if file_path.exists():
                    ts = datetime.fromisoformat(row['timestamp']).strftime('%Y-%m-%d %H:%M')
                    with open(file_path, 'rb') as f:
                        st.sidebar.download_button(
                            f"⬇️ {ts}",
                            f.read(),
                            file_name=file_path.name,
                            key=f"download_{file_path.stem}"
                        )
    
    st.sidebar.divider()
    if st.sidebar.button("🗑️ Réinitialiser le leaderboard", type="primary"):
        conn = get_conn()
        with db_lock:
            conn.execute("DELETE FROM submissions")
            conn.commit()
        st.sidebar.success("Leaderboard réinitialisé")
        st.cache_data.clear()
        time.sleep(0.5)
        st.rerun()
else:
    reveal_private = False
    if admin_mode:
        st.sidebar.warning("Mot de passe admin requis")

# Validate ground once we have it
if ground_df is not None:
    try:
        validate_ground_columns(ground_df)
    except Exception as e:
        st.sidebar.error(f"Ground invalide: {e}")
        ground_df = None

# =============================================================================
# MAIN
# =============================================================================
st.title(COMPETITION_NAME)

# Competition status banner
comp_status = check_competition_status()
if comp_status == "not_started":
    st.warning(f"⏳ La compétition commence le {COMPETITION_START.strftime('%d/%m/%Y')}")
elif comp_status == "ended":
    st.info(f"✅ Compétition terminée le {COMPETITION_END.strftime('%d/%m/%Y')}. Leaderboard final affiché.")
else:
    time_left = COMPETITION_END - datetime.now()
    days_left = time_left.days
    if days_left > 0:
        st.success(f"🏁 Compétition en cours - Il reste {days_left} jour(s)")
    else:
        hours_left = time_left.seconds // 3600
        st.success(f"🏁 Compétition en cours - Il reste {hours_left} heure(s)")

st.write(
    "Téléchargez votre `submission.csv` au format **`id,income_prob`**. "
    "Les scores sont calculés contre des labels cachés (ground). "
    "Chaque soumission est **archivée**."
)
st.info(
    "Métrique principale: **ROC AUC**. Tiebreaker: **Log Loss**. "
    "Un split **Public/Private** est appliqué si `public_ids.csv` est fourni.",
    icon="ℹ️"
)

colL, colR = st.columns([1, 1])

with colL:
    st.header("📤 Soumettre un fichier")
    
    # Check if competition allows submissions
    can_submit = comp_status == "active" or admin_ok
    
    if not can_submit and comp_status == "not_started":
        st.warning("Les soumissions ne sont pas encore ouvertes.")
    elif not can_submit and comp_status == "ended":
        st.warning("Les soumissions sont fermées.")
    
    team = st.text_input("Nom d'équipe (obligatoire)", placeholder="Ex: Team Alpha")
    uploaded_file = st.file_uploader("Choisir votre submission.csv", type=["csv"])
    
    # Show submission count
    if team.strip():
        n_today = count_team_submissions_today(team.strip())
        remaining = MAX_SUBMISSIONS_PER_TEAM_PER_DAY - n_today
        st.caption(f"Soumissions restantes aujourd'hui: {remaining}/{MAX_SUBMISSIONS_PER_TEAM_PER_DAY}")
    
    disabled = ground_df is None or uploaded_file is None or team.strip() == "" or not can_submit
    submit_btn = st.button("Évaluer & publier au leaderboard", type="primary", disabled=disabled)

    if ground_df is None:
        st.warning("Aucun **ground.csv** disponible. Demandez à l'enseignant de l'ajouter en mode Admin.")

    if submit_btn:
        try:
            # File size check
            uploaded_file.seek(0, 2)  # Seek to end
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)  # Reset
            
            if file_size > MAX_FILE_SIZE:
                st.error(f"❌ Fichier trop volumineux. Taille max: {MAX_FILE_SIZE/(1024*1024):.1f} MB")
                st.stop()

            # Rate limiting
            n_today = count_team_submissions_today(team.strip())
            if n_today >= MAX_SUBMISSIONS_PER_TEAM_PER_DAY:
                st.error(f"❌ Limite de soumissions atteinte pour aujourd'hui ({MAX_SUBMISSIONS_PER_TEAM_PER_DAY}).")
                st.stop()

            # Read + validate
            sub = read_uploaded_csv(uploaded_file)
            validate_submission_columns(sub)
            validate_probs(sub["income_prob"])

            # Archive uploaded file early
            saved_path = archive_submission(team.strip(), uploaded_file)

            # Align and score
            df = align_by_id(ground_df, sub)
            overall = compute_scores(df)

            # Public/private split if provided
            pub_auc, priv_auc = None, None
            n_pub, n_priv = None, None
            if public_ids is not None:
                pub_df, priv_df = split_public_private(df, public_ids)
                if pub_df is not None and priv_df is not None:
                    pub_scores = compute_scores(pub_df)
                    priv_scores = compute_scores(priv_df)
                    pub_auc, priv_auc = pub_scores["roc_auc"], priv_scores["roc_auc"]
                    n_pub, n_priv = pub_scores["n"], priv_scores["n"]

            row = {
                "team": team.strip(),
                "filename": saved_path.as_posix(),
                "timestamp": datetime.utcnow().isoformat(),
                "public_auc": pub_auc if pub_auc is not None else overall["roc_auc"],
                "private_auc": priv_auc if priv_auc is not None else None,
                "overall_auc": overall["roc_auc"],
                "logloss": overall["log_loss"],
                "n_overall": overall["n"],
                "n_public": n_pub if n_pub is not None else None,
                "n_private": n_priv if n_priv is not None else None,
            }

            conn = get_conn()
            insert_submission(conn, row)

            st.success("✅ Soumission enregistrée avec succès!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ROC AUC (overall)", f"{overall['roc_auc']:.4f}")
            with col2:
                if pub_auc is not None:
                    st.metric("Public AUC", f"{pub_auc:.4f}")
            with col3:
                if priv_auc is not None and (reveal_private or admin_ok):
                    st.metric("Private AUC", f"{priv_auc:.4f}")
            
            st.caption(f"Log Loss: {overall['log_loss']:.5f} • N={overall['n']}")

            # Refresh leaderboard
            st.cache_data.clear()
            time.sleep(0.5)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erreur lors de l'évaluation: {e}")

with colR:
    st.header("🏆 Leaderboard")
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🥇 Meilleure par équipe", "📊 Toutes les soumissions", "📈 Statistiques"])
    
    conn = get_conn()
    
    # Determine sort order
    if reveal_private and public_ids is not None:
        order_by = "private_auc DESC, logloss ASC, timestamp ASC"
        st.info("🔓 Leaderboard privé révélé!", icon="🔓")
    else:
        order_by = "public_auc DESC, overall_auc DESC, logloss ASC, timestamp ASC"
    
    with db_lock:
        lb = pd.read_sql_query(
            f"SELECT * FROM submissions ORDER BY {order_by}",
            conn
        )
    
    with tab1:
        if not lb.empty:
            # Best submission per team
            best_per_team = lb.sort_values(['team', 'public_auc'], ascending=[True, False])
            best_per_team = best_per_team.groupby('team').first().reset_index()
            
            # Re-sort by score
            if reveal_private and 'private_auc' in best_per_team.columns and best_per_team['private_auc'].notna().any():
                best_per_team = best_per_team.sort_values('private_auc', ascending=False)
            else:
                best_per_team = best_per_team.sort_values('public_auc', ascending=False)
            
            best_per_team['rank'] = range(1, len(best_per_team) + 1)
            best_per_team['timestamp'] = pd.to_datetime(best_per_team['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            if not reveal_private and 'private_auc' in best_per_team.columns:
                best_per_team['private_auc'] = '---'
            
            display_cols = ["rank", "team", "public_auc", "private_auc", "logloss", "timestamp"]
            display_cols = [c for c in display_cols if c in best_per_team.columns]
            
            st.dataframe(best_per_team[display_cols], use_container_width=True, height=400)
        else:
            st.write("Aucune soumission.")
    
    with tab2:
        if lb.empty:
            st.write("Aucune soumission pour le moment.")
        else:
            show = lb.copy()
            show = show.drop(columns=["sid"])
            
            # Format timestamp
            show['timestamp'] = pd.to_datetime(show['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            
            # Add rank
            show.insert(0, 'rank', range(1, len(show) + 1))
            
            # Hide private scores if not revealed
            if not reveal_private and 'private_auc' in show.columns:
                show['private_auc'] = '---'
            
            cols = ["rank", "team", "timestamp", "public_auc", "private_auc", "overall_auc", "logloss", "n_overall", "n_public", "n_private"]
            cols = [c for c in cols if c in show.columns]
            show = show[cols]
            
            st.dataframe(show, use_container_width=True, height=400)

            csv = show.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger (CSV)", csv, file_name="leaderboard.csv", mime="text/csv")
    
    with tab3:
        if not lb.empty:
            st.subheader("📊 Statistiques par équipe")
            team_stats = lb.groupby('team').agg({
                'sid': 'count',
                'public_auc': 'max',
                'logloss': 'min',
                'timestamp': 'max'
            }).rename(columns={
                'sid': 'Nb soumissions',
                'public_auc': 'Meilleur AUC',
                'logloss': 'Meilleur Log Loss',
                'timestamp': 'Dernière soumission'
            })
            team_stats['Dernière soumission'] = pd.to_datetime(team_stats['Dernière soumission']).dt.strftime('%Y-%m-%d %H:%M')
            team_stats = team_stats.sort_values('Meilleur AUC', ascending=False)
            st.dataframe(team_stats, use_container_width=True)
            
            st.subheader("📈 Distribution des scores")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Nombre total de soumissions", len(lb))
                st.metric("Nombre d'équipes", lb['team'].nunique())
            with col2:
                st.metric("Meilleur AUC", f"{lb['public_auc'].max():.4f}")
                st.metric("AUC moyen", f"{lb['public_auc'].mean():.4f}")

st.divider()
with st.expander("📘 Format attendu des fichiers"):
    st.markdown(f"""
**Submission** – `submission.csv`  

Le fichier doit contenir exactement 2 colonnes :
- `id` : identifiant de chaque observation (entier)
- `income_prob` : probabilité que l'individu gagne >50K (float entre 0 et 1)

Exemple :
```
id,income_prob
1,0.75
2,0.32
3,0.89
...
```

**Notes importantes :**
- Le fichier ne doit pas contenir de valeurs manquantes
- Les probabilités doivent être comprises entre 0 et 1
- Tous les IDs du test set doivent être présents
- Taille maximale du fichier : {MAX_FILE_SIZE/(1024*1024):.0f} MB
""")

st.caption(f"© Mini-Kaggle Platform v2.0 – Compétition: {COMPETITION_START.strftime('%d/%m/%Y')} → {COMPETITION_END.strftime('%d/%m/%Y')}")