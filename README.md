# 🧬 Neuro-Hepato Screener

## ⚖️ License
**The MIT License (MIT)** Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the Software), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---

## 🚀 Project Overview

**Neuro-Hepato Screener** is a hybrid AI tool designed to solve the **"Physicochemical Paradox"** in early-stage drug discovery.

Compounds designed to treat brain disorders must cross the **Blood-Brain Barrier (BBB)**, which often requires high lipophilicity. However, this same property significantly increases the risk of **Drug-Induced Liver Injury (DILI)**. This tool helps researchers identify candidates that are both **permeable** and **safe** before expensive synthesis begins.

### 📄 Deep Dive
For a detailed technical explanation of the architecture, algorithms, and validation case studies (including the Amiodarone test), please read the full research paper:

👉 **[Download the Technical White Paper (PDF)](./white_paper.pdf)**

---

## 💡 How It Works
The system utilizes a **Parallel Inference Architecture** that runs two distinct validation layers simultaneously:

1.  **The AI Layer (Probabilistic):**
    * Runs two Random Forest Classifiers trained on **Tox21** (Liver Toxicity) and **B3DB** (Brain Permeability) datasets.
    * Predicts biological activity based on structural fingerprints.

2.  **The Explainability Layer (Deterministic):**
    * Calculates a **CNS MPO (Multi-Parameter Optimization) Score** using RDKit.
    * Evaluates physicochemical properties (LogP, TPSA, MW, HBD) against strict medicinal chemistry rules.
    * Acts as a safety valve to catch false negatives from the AI models.

🔗 **Live App:** https://neuro-hepato-screen.streamlit.app/

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Cheminformatics:** RDKit
* **Deployment:** Streamlit Cloud


