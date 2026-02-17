"""
NEURO-HEPATO SCREENER - EXACT ORIGINAL INTERFACE
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Neuro-Hepato Screener",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    .block-container {
        padding-top: 2rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
   
    
    /* CNS Score Box */
    .cns-box {
        background: transparent;
        padding: 1rem 0;
    }
    
    .cns-label {
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .cns-value {
        
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }
    
    /* Green success box */
    .success-box {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        border-radius: 12px;
        padding: 1.5rem 2rem;
    }
    
    .success-title {
        color: #4ade80 !important;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }
    
    .success-text {
        color: #d1fae5 !important;
        font-size: 1rem;
        margin: 0;
    }
    
    /* Property cards - transparent background */
    .prop-card {
        text-align: center;
        padding: 0.5rem;
    }
    
    .prop-label {
        
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    .prop-value {
        
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
        margin: 0.5rem 0;
    }
    
    .prop-target {
        color: #4ade80 !important;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Prediction boxes - dark green */
    .pred-box {
        background: linear-gradient(135deg, #1e5128 0%, #2d6a3e 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .pred-title {
        color: #4ade80 !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Input styling */
    .stTextInput input {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    
    .stTextInput label {
    
    }
    
    /* Button */
    .stButton button {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    try:
        liver = joblib.load('models/hepatotoxicity_model.pkl')
        bbb = joblib.load('models/bbb_model.pkl')
        return liver, bbb
    except:
        return None, None

liver_model, bbb_model = load_models()

# ============================================================================
# CALCULATIONS
# ============================================================================

def calculate_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        return {
            'LogP': round(Descriptors.MolLogP(mol), 1),
            'TPSA': round(Descriptors.TPSA(mol), 1),
            'MW': round(Descriptors.MolWt(mol), 0),
            'HBD': Descriptors.NumHDonors(mol)
        }
    except:
        return None

def calculate_cns_mpo(props):
    scores = []
    
    # LogP
    if 1 <= props['LogP'] <= 3:
        scores.append(1.0)
    elif props['LogP'] < 5:
        scores.append(0.5)
    else:
        scores.append(0.0)
    
    # TPSA
    if 40 <= props['TPSA'] <= 90:
        scores.append(1.0)
    elif props['TPSA'] < 40:
        scores.append(0.75)
    else:
        scores.append(0.25)
    
    # MW
    if props['MW'] < 360:
        scores.append(1.0)
    elif props['MW'] < 500:
        scores.append(0.5)
    else:
        scores.append(0.0)
    
    # HBD
    if props['HBD'] == 0:
        scores.append(1.0)
    elif props['HBD'] == 1:
        scores.append(0.75)
    else:
        scores.append(0.25)
    
    scores.extend([0.75, 0.75])  # placeholders
    
    return round(sum(scores), 1)

def create_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_array = np.array(fp)
        
        desc = np.array([
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol)
        ])
        
        return np.concatenate([fp_array, desc]).reshape(1, -1)
    except:
        return None

# ============================================================================
# ============================================================================

st.markdown("""
# 🧬 Neuro-Hepato Screener

**An AI-Driven Safety Assessment Tool for the Liver-Brain Axis.**

Developing drugs for brain disorders is difficult because compounds that cross the **Blood-Brain Barrier (BBB)** are often too lipophilic, leading to **Drug-Induced Liver Injury (DILI)**.

This tool solves that problem by calculating a **CNS MPO (Multi-Parameter Optimization) Score**, helping researchers identify candidates that are both **permeable** and **safe**.
""")

st.markdown("---")

# ============================================================================
# ============================================================================

st.markdown("## Run Analysis")

st.markdown("**Enter SMILES String:**")

smiles_input = st.text_input(
    "SMILES",
    value="",
    placeholder="e.g., CCCCC1=C(C(=O)c2ccccc2O1)c2ccccc2O1",
    label_visibility="collapsed"
)

if st.button("Analyze Candidate"):
    
    if not smiles_input:
        st.warning("⚠️ Please enter a SMILES string.")
    
    elif liver_model is None or bbb_model is None:
        st.error("❌ Models not found.")
    
    else:
        props = calculate_properties(smiles_input)
        
        if props is None:
            st.error("❌ Invalid SMILES string.")
        else:
            cns_mpo = calculate_cns_mpo(props)
            features = create_features(smiles_input)
            
            if features is not None:
                liver_prob = liver_model.predict_proba(features)[0]
                bbb_prob = bbb_model.predict_proba(features)[0]
                liver_pred = liver_model.predict(features)[0]
                bbb_pred = bbb_model.predict(features)[0]
                
                # ============================================================
                # RESULTS - EXACTLY AS IN SCREENSHOT
                # ============================================================
                
                st.markdown("---")
                
                # CNS MPO + Success Box (2 columns)
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"""
                    <div class="cns-box">
                        <div class="cns-label">CNS MPO Score</div>
                        <div class="cns-value">{cns_mpo}/6.0</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    safe_prob = liver_prob[0]       # probability of being SAFE
                    permeable_prob = bbb_prob[1]    # probability of being BBB+

                    if safe_prob > 0.7 and permeable_prob > 0.7 and cns_mpo >= 4.0:
                        st.markdown("""
                        <div class="success-box">
                            <div class="success-title">⭐ High Potential Candidate</div>
                            <div class="success-text">Optimal balance of safety and permeability.</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif safe_prob < 0.5:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
                                    border-radius: 12px; padding: 1.5rem 2rem;">
                            <div style="color: #fb923c; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
                                ⚠️ Caution: Hepatotoxicity Risk
                            </div>
                            <div style="color: #fed7aa; font-size: 1rem;">
                                Potential liver toxicity detected. Consider structural modifications.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif permeable_prob < 0.3:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%);
                                    border-radius: 12px; padding: 1.5rem 2rem;">
                            <div style="color: #93c5fd; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
                                🛡️ Low CNS Penetration
                            </div>
                            <div style="color: #bfdbfe; font-size: 1rem;">
                                Compound unlikely to cross the blood-brain barrier.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #374151 0%, #4b5563 100%);
                                    border-radius: 12px; padding: 1.5rem 2rem;">
                            <div style="color: #d1d5db; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem;">
                                🔧 Needs Optimization
                            </div>
                            <div style="color: #9ca3af; font-size: 1rem;">
                                Shows some promise but requires further refinement.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Molecular Properties
                st.markdown("## 🔬 Molecular Properties")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    target = "↑ Target: 1-3" if 1 <= props['LogP'] <= 3 else ""
                    st.markdown(f"""
                    <div class="prop-card">
                        <div class="prop-label">LogP (Oiliness)</div>
                        <div class="prop-value">{props['LogP']}</div>
                        <div class="prop-target">{target}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    target = "↑ Target: 40-90" if 40 <= props['TPSA'] <= 90 else ""
                    st.markdown(f"""
                    <div class="prop-card">
                        <div class="prop-label">TPSA (Polarity)</div>
                        <div class="prop-value">{props['TPSA']}</div>
                        <div class="prop-target">{target}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    target = "↑ Target: <360" if props['MW'] < 360 else ""
                    st.markdown(f"""
                    <div class="prop-card">
                        <div class="prop-label">MW (Size)</div>
                        <div class="prop-value">{int(props['MW'])}</div>
                        <div class="prop-target">{target}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    target = "↑ Target: 0-1" if props['HBD'] <= 1 else ""
                    st.markdown(f"""
                    <div class="prop-card">
                        <div class="prop-label">HBD (Stickiness)</div>
                        <div class="prop-value">{props['HBD']}</div>
                        <div class="prop-target">{target}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # AI Prediction Models
                st.markdown("## 🤖 AI Prediction Models")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if liver_pred == 0:
                        conf = liver_prob[0] * 100
                        st.markdown(f"""
                        <div class="pred-box">
                            <div class="pred-title">✓ Low Liver Toxicity Risk ({conf:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        conf = liver_prob[1] * 100
                        st.markdown(f"""
                        <div class="pred-box">
                            <div class="pred-title">⚠️ High Liver Toxicity Risk ({conf:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    if bbb_pred == 1:
                        conf = bbb_prob[1] * 100
                        st.markdown(f"""
                        <div class="pred-box">
                            <div class="pred-title">🧠 High Brain Permeability ({conf:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        conf = bbb_prob[0] * 100
                        st.markdown(f"""
                        <div class="pred-box">
                            <div class="pred-title">🛡️ Low Brain Permeability ({conf:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)


