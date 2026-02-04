import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier

# Page Config
st.set_page_config(page_title="Liver-Brain Safety Screen", layout="centered")

st.title("🧠 Liver-Brain Axis Safety Screen")
st.markdown("""
**Prototype v1.0** | Built for Early Safety Assessment
This tool screens compounds for **Hepatotoxicity (DILI)** and **Blood-Brain Barrier (BBB) Permeability**.
It uses a sequential Random Forest architecture.
""")

# --- 1. DATA LOADING (Cached) ---
@st.cache_data
def load_data():
    try:
        # Load the CSVs you downloaded
        tox_df = pd.read_csv("tox21_data.csv") 
        bbb_df = pd.read_csv("b3db_data.csv")   
        return tox_df, bbb_df
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please upload 'tox21_data.csv' and 'b3db_data.csv' to your GitHub.")
        return None, None

# --- 2. FEATURIZATION ---
def get_fingerprint(smiles, radius=2, n_bits=2048):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None

# --- 3. MODEL TRAINING (Cached) ---
@st.cache_resource
def train_models(tox_df, bbb_df):
    # Train Liver Model
    tox_df['fp'] = tox_df['smiles'].apply(get_fingerprint)
    tox_data = tox_df.dropna(subset=['fp'])
    X_liver = np.stack(tox_data['fp'].values)
    y_liver = tox_data['activity'].values 
    
    clf_liver = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_liver.fit(X_liver, y_liver)
    
    # Train BBB Model
    bbb_df['fp'] = bbb_df['smiles'].apply(get_fingerprint)
    bbb_data = bbb_df.dropna(subset=['fp'])
    X_bbb = np.stack(bbb_data['fp'].values)
    # Convert B3DB labels to 0/1
    y_bbb = bbb_data['BBB+/BBB-'].apply(lambda x: 1 if x == 'BBB+' else 0).values 
    
    clf_bbb = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_bbb.fit(X_bbb, y_bbb)
    
    return clf_liver, clf_bbb

# --- 4. THE APP LOGIC ---
tox_df, bbb_df = load_data()

if tox_df is not None:
    with st.spinner('Initializing AI models... (this takes 30 seconds)'):
        try:
            liver_model, bbb_model = train_models(tox_df, bbb_df)
            st.success("✅ System Online")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            liver_model = None

    # User Input
    st.divider()
    smiles_input = st.text_input("Enter SMILES String:", placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O")
    
    if st.button("Analyze Compound"):
        if not smiles_input:
            st.warning("Please enter a SMILES string.")
        elif liver_model:
            fp = get_fingerprint(smiles_input)
            if fp is not None:
                fp_reshaped = fp.reshape(1, -1)
                
                # Prediction 1: Liver
                liver_prob = liver_model.predict_proba(fp_reshaped)[0][1]
                liver_pred = "TOXIC" if liver_prob > 0.5 else "SAFE"
                
                # Prediction 2: BBB
                bbb_prob = bbb_model.predict_proba(fp_reshaped)[0][1]
                bbb_pred = "CROSSES BBB" if bbb_prob > 0.5 else "DOES NOT CROSS"
                
                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Liver Toxicity")
                    if liver_pred == "TOXIC":
                        st.error(f"⚠️ High Risk ({liver_prob:.2f})")
                    else:
                        st.success(f"✅ Low Risk ({liver_prob:.2f})")
                
                with col2:
                    st.subheader("Brain Permeability")
                    if bbb_pred == "CROSSES BBB":
                        st.warning(f"🧠 High Permeability ({bbb_prob:.2f})")
                    else:
                        st.info(f"🛡️ Low Permeability ({bbb_prob:.2f})")
                
                st.info(f"**Interpretation:** {liver_pred} to Liver, {bbb_pred}")
            else:
                st.error("Invalid SMILES string.")