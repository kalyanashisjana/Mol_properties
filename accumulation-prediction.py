# ==========================================================
# Correlation and Prediction of Accumulation with Globularity
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------
# 1. Load data
# -------------------------------
files = [
    "blactam_input.csv",
    "blactam_abx-prediction.csv"
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["SourceFile"] = f  # keep track of origin
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Ensure numeric types
for col in ["Molecular_Weight", "Rotatable_Bonds", "Globularity", "ACCUMULATION"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print("Merged data shape:", df.shape)

# -------------------------------
# 2. Add charge features from SMILES
# -------------------------------
def get_charge_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.nan, "Invalid"

    net_charge = Chem.GetFormalCharge(mol)
    pos_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() > 0)
    neg_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetFormalCharge() < 0)

    if net_charge > 0:
        charge_type = "Positive"
    elif net_charge < 0:
        charge_type = "Negative"
    elif pos_atoms > 0 and neg_atoms > 0:
        charge_type = "Zwitterion"
    else:
        charge_type = "Neutral"

    return net_charge, charge_type

charges = df["SMILES"].apply(get_charge_features)
df["Net_Charge"] = [c[0] for c in charges]
df["Charge_Type"] = [c[1] for c in charges]

# -------------------------------
# 3. Split Set A (with data) and Set B (without)
# -------------------------------
setA = df[df["ACCUMULATION"].notnull()].copy()
setB = df[df["ACCUMULATION"].isnull()].copy()

print(f"Set A size = {len(setA)}")
print(f"Set B size = {len(setB)}")

# -------------------------------
# 4. Explore correlations
# -------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(data=setA, x="Globularity", y="ACCUMULATION", hue="Charge_Type", style="Charge_Type", s=80)
plt.title("Accumulation vs Globularity (Set A)")
plt.savefig("Accumulation_vs_Globularity-seta.png")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(data=setA, x="Rotatable_Bonds", y="ACCUMULATION", hue="Charge_Type", s=80)
plt.title("Accumulation vs Rotatable Bonds (Set A)")
plt.savefig("Accumulation_vs_Rotatable-setb.png")
plt.show()

# Correlation heatmap (numeric only)
corr = setA[["ACCUMULATION","Globularity","Rotatable_Bonds","Molecular_Weight","Net_Charge"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix (Set A)")
plt.savefig("Correlation-Matrix-seta.png")
plt.show()

# -------------------------------
# 5. Regression Model
# -------------------------------
X = setA[["Globularity","Rotatable_Bonds","Net_Charge"]].fillna(0)
y = setA["ACCUMULATION"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("R2 score on test:", r2_score(y_test,y_pred))
print("RMSE on test:", np.sqrt(mean_squared_error(y_test,y_pred)))
print("Coefficients:", dict(zip(X.columns, model.coef_)))
print("Intercept:", model.intercept_)

# -------------------------------
# 6. Predict Set B
# -------------------------------
if not setB.empty:
    Xb = setB[["Globularity","Rotatable_Bonds","Net_Charge"]].fillna(0)
    setB["Predicted_ACCUMULATION"] = model.predict(Xb)

    # Merge results
    df_out = pd.concat([setA, setB])
    df_out.to_csv("accumulation_with_predictions.csv", index=False)
    print("Predictions saved to accumulation_with_predictions.csv")
    print(setB[["SMILES","Globularity","Rotatable_Bonds","Net_Charge","Charge_Type","Predicted_ACCUMULATION"]].head())

