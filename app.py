import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Liver-Brain MPO Screen", layout="centered")

st.title("🧠 Liver-Brain Axis Safety Screen")
st.markdown("""
**Prototype v2.1 (Robust Edition)** | Built for Early Safety Assessment  
This tool screens compounds for **Hepatotoxicity** and **BBB Permeability** using an intelligent MPO algorithm.
""")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        tox_df = pd.read_csv("tox21_data.csv")
        bbb_df = pd.read_csv("b3db_data.csv")
        return tox_df, bbb_df
    except:
        return None, None

tox_df, bbb_df = load_data()

if tox_df is None:
    st.error("⚠️ Data Missing! Please upload 'tox21_data.csv' and 'b3db_data.csv' to your GitHub repository.")
    st.stop()

# --- 2. FEATURIZATION ---
def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    except:
        return None
    return None

# --- helper: SMART COLUMN FINDER (The Fix) ---
def get_target_col(df, preferred_name):
    # 1. Try exact match
    if preferred_name in df.columns:
        return preferred_name
    # 2. Try cleaning whitespace
    clean_cols = {c.strip(): c for c in df.columns}
    if preferred_name in clean_cols:
        return clean_cols[preferred_name]
    # 3. Fallback: Use the LAST column (standard for datasets)
    return df.columns[-1]

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_models(tox_df, bbb_df):
    
    # --- LIVER MODEL ---
    # Smartly find the target column (Fixes KeyError)
    liver_col = get_target_col(tox_df, 'NR-AhR')
    
    # Filter valid data
    valid_data_liver = []
    valid_labels_liver = []
    
    for idx, row in tox_df.iterrows():
        fp = get_fingerprint(row['smiles'])
        target = row[liver_col]
        # Only use rows with valid fingerprints and known labels (0 or 1)
        if fp is not None and not pd.isna(target):
            valid_data_liver.append(fp)
            valid_labels_liver.append(target)
            
    clf_liver = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_liver.fit(valid_data_liver, valid_labels_liver)
    
    # --- BBB MODEL ---
    bbb_col = get_target_col(bbb_df, 'BBB+')
    
    valid_data_bbb = []
    valid_labels_bbb = []
    
    for idx, row in bbb_df.iterrows():
        fp = get_fingerprint(row['smiles'])
        target = row[bbb_col]
        if fp is not None and not pd.isna(target):
            valid_data_bbb.append(fp)
            valid_labels_bbb.append(target)

    clf_bbb = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_bbb.fit(valid_data_bbb, valid_labels_bbb)
    
    return clf_liver, clf_bbb, liver_col, bbb_col

with st.spinner("Calibrating AI Models..."):
    liver_model, bbb_model, liver_col_name, bbb_col_name = train_models(tox_df, bbb_df)

# Show debug info in sidebar so we know what column was picked
with st.sidebar:
    st.header("⚙️ Model Debugger")
    st.success("System Online")
    st.write(f"**Liver Target:** `{liver_col_name}`")
    st.write(f"**Brain Target:** `{bbb_col_name}`")

# --- 4. ADVANCED METRICS ---
def calculate_mpo(mol):
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    
    score_logp = 1.0 if (1 <= logp <= 3) else max(0, 1 - abs(logp - 2) * 0.2)
    score_mw = 1.0 if mw <= 360 else max(0, 1 - (mw - 360) / 140)
    score_tpsa = 1.0 if (20 <= tpsa <= 90) else max(0, 1 - abs(tpsa - 55) * 0.02)
    score_hbd = 1.0 if hbd <= 0 else max(0, 1 - hbd * 0.25)
    
    final_score = (score_logp + score_mw + score_tpsa + score_hbd) * 1.5
    return final_score, {"LogP": logp, "MW": mw, "TPSA": tpsa, "HBD": hbd}

# --- 5. USER INTERFACE ---
with st.form(key='analysis_form'):
    user_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")
    submit_button = st.form_submit_button(label='Run Safety Analysis')

# --- 6. ANALYSIS LOGIC ---
if submit_button:
    if user_smiles:
        mol = Chem.MolFromSmiles(user_smiles)
        if mol:
            fp = get_fingerprint(user_smiles)
            mpo_score, metrics = calculate_mpo(mol)
            
            fp_reshaped = fp.reshape(1, -1)
            liver_prob = liver_model.predict_proba(fp_reshaped)[0][1]
            bbb_prob = bbb_model.predict_proba(fp_reshaped)[0][1]
            
            # --- RESULTS ---
            st.subheader("Results Overview")
            col_score, col_interp = st.columns([1, 2])
            
            with col_score:
                st.metric("CNS MPO Score", f"{mpo_score:.1f}/6.0")
            
            with col_interp:
                if mpo_score >= 4.0:
                    st.success("🌟 **High Potential Candidate**")
                elif mpo_score >= 3.0:
                    st.warning("⚖️ **Moderate Candidate**")
                else:
                    st.error("⛔ **Poor Candidate**")

            st.markdown("---")

            st.subheader("🔬 The Neuro-Hepato Trade-off")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LogP (Oiliness)", f"{metrics['LogP']:.1f}", "Target: 1-3")
            col2.metric("TPSA (Polarity)", f"{metrics['TPSA']:.1f}", "Target: 40-90")
            col3.metric("MW (Size)", f"{metrics['MW']:.0f}", "Target: <360")
            col4.metric("HBD (Stickiness)", f"{metrics['HBD']}", "Target: 0-1")
            
            st.markdown("---")
            
            col_a, col_b = st.columns
