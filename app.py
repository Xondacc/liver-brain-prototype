import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from sklearn.ensemble import RandomForestClassifier

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Neuro-Hepato Screener", layout="centered")

# --- SIDEBAR (Clean & Professional) ---
with st.sidebar:
    st.header("About Project")
    st.info("""
    **Neuro-Hepato Screener** is an AI-powered tool designed to solve the *Physicochemical Paradox* in drug discovery.
    
    It screens molecules for:
    * 🧠 **Brain Permeability** (B3DB)
    * 🫁 **Liver Safety** (Tox21)
    
    **Version:** 2.1 (Stable)
    """)
    st.markdown("---")
    st.caption("Built with Python, RDKit & Scikit-Learn")

# --- MAIN TITLE & DESCRIPTION ---
st.title("🧬 Neuro-Hepato Screener")
st.markdown("""
**An AI-Driven Safety Assessment Tool for the Liver-Brain Axis.**

Developing drugs for brain disorders is difficult because compounds that cross the **Blood-Brain Barrier (BBB)** are often too lipophilic, leading to **Drug-Induced Liver Injury (DILI)**.

This tool solves that problem by calculating a **CNS MPO (Multi-Parameter Optimization) Score**, helping researchers identify candidates that are both **permeable** and **safe**.
""")

# --- 1. DATA LOADING (Robust) ---
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
    st.error("⚠️ System Error: Data files not found. Please upload 'tox21_data.csv' and 'b3db_data.csv' to GitHub.")
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

# --- helper: SMART COLUMN FINDER ---
def get_target_col(df, preferred_name):
    if preferred_name in df.columns:
        return preferred_name
    clean_cols = {c.strip(): c for c in df.columns}
    if preferred_name in clean_cols:
        return clean_cols[preferred_name]
    return df.columns[-1]

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_models(tox_df, bbb_df):
    
    # Liver Model
    liver_col = get_target_col(tox_df, 'NR-AhR')
    valid_data_liver = []
    valid_labels_liver = []
    
    for idx, row in tox_df.iterrows():
        fp = get_fingerprint(row['smiles'])
        target = row[liver_col]
        if fp is not None and not pd.isna(target):
            valid_data_liver.append(fp)
            valid_labels_liver.append(target)
            
    clf_liver = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_liver.fit(valid_data_liver, valid_labels_liver)
    
    # BBB Model
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
    
    return clf_liver, clf_bbb

with st.spinner("Initializing AI Engines..."):
    liver_model, bbb_model = train_models(tox_df, bbb_df)

# --- 4. ADVANCED METRICS ---
def calculate_mpo(mol):
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    
    # Pfizer CNS MPO Scoring Rules
    score_logp = 1.0 if (1 <= logp <= 3) else max(0, 1 - abs(logp - 2) * 0.2)
    score_mw = 1.0 if mw <= 360 else max(0, 1 - (mw - 360) / 140)
    score_tpsa = 1.0 if (20 <= tpsa <= 90) else max(0, 1 - abs(tpsa - 55) * 0.02)
    score_hbd = 1.0 if hbd <= 0 else max(0, 1 - hbd * 0.25)
    
    final_score = (score_logp + score_mw + score_tpsa + score_hbd) * 1.5
    return final_score, {"LogP": logp, "MW": mw, "TPSA": tpsa, "HBD": hbd}

# --- 5. USER INTERFACE ---
with st.form(key='analysis_form'):
    st.subheader("Run Analysis")
    user_smiles = st.text_input("Enter SMILES String:", "CC(=O)OC1=CC=CC=C1C(=O)O")
    submit_button = st.form_submit_button(label='Analyze Candidate')

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
            
            # --- RESULTS DASHBOARD ---
            st.markdown("---")
            
            # A. The Executive Summary
            col_score, col_interp = st.columns([1, 2])
            
            with col_score:
                st.metric("CNS MPO Score", f"{mpo_score:.1f}/6.0")
            
            with col_interp:
                if mpo_score >= 4.0:
                    st.success("🌟 **High Potential Candidate**\n\nOptimal balance of safety and permeability.")
                elif mpo_score >= 3.0:
                    st.warning("⚖️ **Moderate Candidate**\n\nUsable, but check LogP and TPSA values.")
                else:
                    st.error("⛔ **Poor Candidate**\n\nHigh risk of failure due to toxicity or poor entry.")

            st.markdown("---")

            # B. The Mechanics
            st.subheader("🔬 Molecular Properties")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LogP (Oiliness)", f"{metrics['LogP']:.1f}", "Target: 1-3")
            col2.metric("TPSA (Polarity)", f"{metrics['TPSA']:.1f}", "Target: 40-90")
            col3.metric("MW (Size)", f"{metrics['MW']:.0f}", "Target: <360")
            col4.metric("HBD (Stickiness)", f"{metrics['HBD']}", "Target: 0-1")
            
            # C. AI Predictions
            st.markdown("### 🤖 AI Prediction Models")
            col_a, col_b = st.columns(2)
            with col_a:
                if liver_prob > 0.5:
                    st.error(f"⚠️ **High Liver Toxicity Risk** ({liver_prob:.1%})")
                else:
                    st.success(f"✅ **Low Liver Toxicity Risk** ({liver_prob:.1%})")
            
            with col_b:
                if bbb_prob > 0.5:
                    st.success(f"🧠 **High Brain Permeability** ({bbb_prob:.1%})")
                else:
                    st.warning(f"🛡️ **Low Brain Permeability** ({bbb_prob:.1%})")

        else:
            st.error("Invalid SMILES string. Please check the structure.")
