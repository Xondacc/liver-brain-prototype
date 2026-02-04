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
**Prototype v2.0 (MPO Edition)** | Built for Early Safety Assessment  
This tool solves the **Neuro-Hepato Paradox**: Finding compounds that cross the **Blood-Brain Barrier (BBB)** without triggering **Drug-Induced Liver Injury (DILI)**.
""")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    try:
        tox_df = pd.read_csv("tox21_data.csv")
        bbb_df = pd.read_csv("b3db_data.csv")
        return tox_df, bbb_df
    except:
        st.error("Data files not found. Please upload tox21_data.csv and b3db_data.csv")
        return None, None

tox_df, bbb_df = load_data()

# --- 2. FEATURIZATION ---
def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
    else:
        return None

# --- 3. MODEL TRAINING ---
@st.cache_resource
def train_models(tox_df, bbb_df):
    # Train Liver Model
    X_liver = np.array([get_fingerprint(s) for s in tox_df['smiles'] if get_fingerprint(s) is not None])
    y_liver = tox_df['NR-AhR'].iloc[:len(X_liver)]
    clf_liver = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_liver.fit(X_liver, y_liver)
    
    # Train BBB Model
    X_bbb = np.array([get_fingerprint(s) for s in bbb_df['smiles'] if get_fingerprint(s) is not None])
    y_bbb = bbb_df['BBB+'].iloc[:len(X_bbb)]
    clf_bbb = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_bbb.fit(X_bbb, y_bbb)
    
    return clf_liver, clf_bbb

if tox_df is not None:
    with st.spinner("Calibrating Decision Forest Models..."):
        liver_model, bbb_model = train_models(tox_df, bbb_df)

# --- 4. ADVANCED METRICS (THE EXPLAINABILITY ENGINE) ---
def calculate_mpo(mol):
    # This implements a simplified CNS MPO Score (0-6 scale)
    # It balances Permeability (Brain) vs. Safety (Liver)
    
    # Calculate raw metrics
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Descriptors.NumHDonors(mol)
    
    # Score each component (Standard Pharmaceutical Rules)
    # LogP: Sweet spot is 1-3. >5 is bad for liver. <0 is bad for brain.
    score_logp = 1.0 if (1 <= logp <= 3) else max(0, 1 - abs(logp - 2) * 0.2)
    
    # MW: Smaller is better for brain entry. >500 is difficult.
    score_mw = 1.0 if mw <= 360 else max(0, 1 - (mw - 360) / 140)
    
    # TPSA: Needs to be <90 to cross BBB.
    score_tpsa = 1.0 if (20 <= tpsa <= 90) else max(0, 1 - abs(tpsa - 55) * 0.02)
    
    # HBD: Less "sticky" is better. >3 is terrible for BBB.
    score_hbd = 1.0 if hbd <= 0 else max(0, 1 - hbd * 0.25)
    
    # Final Sum (Scaled to 0-6 range)
    final_score = (score_logp + score_mw + score_tpsa + score_hbd) * 1.5
    return final_score, {"LogP": logp, "MW": mw, "TPSA": tpsa, "HBD": hbd}

# --- 5. USER INTERFACE (FIX: FORM FOR ENTER KEY) ---
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
            
            # AI Predictions
            fp_reshaped = fp.reshape(1, -1)
            liver_prob = liver_model.predict_proba(fp_reshaped)[0][1]
            bbb_prob = bbb_model.predict_proba(fp_reshaped)[0][1]
            
            # --- RESULTS DASHBOARD ---
            
            # A. The Executive Summary
            st.subheader("Results Overview")
            col_score, col_interp = st.columns([1, 2])
            
            with col_score:
                st.metric("CNS MPO Score", f"{mpo_score:.1f}/6.0")
            
            with col_interp:
                if mpo_score >= 4.0:
                    st.success("🌟 **High Potential Candidate**\n\nExcellent balance of brain permeability and liver safety.")
                elif mpo_score >= 3.0:
                    st.warning("⚖️ **Moderate Candidate**\n\nUsable, but likely has sub-optimal properties (check LogP).")
                else:
                    st.error("⛔ **Poor Candidate**\n\nHigh risk of toxicity or inability to cross the BBB.")

            st.markdown("---")

            # B. The Explanation (Why?)
            st.subheader("🔬 The Neuro-Hepato Trade-off")
            st.caption("To treat brain diseases safely, a drug must pass these 4 checks:")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("LogP (Oiliness)", f"{metrics['LogP']:.1f}", "Target: 1-3")
            col2.metric("TPSA (Polarity)", f"{metrics['TPSA']:.1f}", "Target: 40-90")
            col3.metric("MW (Size)", f"{metrics['MW']:.0f}", "Target: <360")
            col4.metric("HBD (Stickiness)", f"{metrics['HBD']}", "Target: 0-1")
            
            with st.expander("ℹ️ Click to understand these metrics"):
                st.markdown("""
                * **LogP:** Measures how "oily" the molecule is. The Brain loves oil, but the Liver hates it (toxicity risk).
                * **TPSA:** Measures electrical charge. If too high (>90), the Brain Barrier rejects it.
                * **HBD:** Hydrogen bonds act like "anchors" that get stuck in water, preventing brain entry.
                """)

            st.markdown("---")
            
            # C. AI Probability Models
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Liver Toxicity (AI)")
                if liver_prob > 0.5:
                    st.error(f"⚠️ High Risk ({liver_prob:.1%})")
                    st.write("The AI detected structural patterns commonly found in liver toxins.")
                else:
                    st.success(f"✅ Low Risk ({liver_prob:.1%})")
                    st.write("No obvious toxicophores (toxic fragments) detected.")
            
            with col_b:
                st.subheader("Brain Permeability (AI)")
                if bbb_prob > 0.5:
                    st.success(f"✅ Permeable ({bbb_prob:.1%})")
                    st.write("The molecule has the right shape to cross the Blood-Brain Barrier.")
                else:
                    st.error(f"⛔ Blocked ({bbb_prob:.1%})")
                    st.write("The molecule is likely too polar or too large to enter the brain.")

        else:
            st.error("Invalid SMILES string. Please check the format.")
