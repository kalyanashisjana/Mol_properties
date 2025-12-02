import csv
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

input_file = "blactam.csv"
output_file = "molecular_properties_with_globularity1.csv"

def calculate_globularity(coordinates):
    if coordinates.shape[0] == 0:
        return None
    centered = coordinates - np.mean(coordinates, axis=0)
    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if min(eigenvalues) <= 0:
        return None
    condition_number = max(eigenvalues) / min(eigenvalues)
    return 1.0 / condition_number if condition_number != 0 else None

with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile)

    # Write header
    writer.writerow(["SMILES", "ACCUMULATION", "Molecular_Weight", "Rotatable_Bonds", "Globularity"])

    for i, row in enumerate(reader):
        smiles = row["SMILES"].strip()
        accumulation = row.get("ACCUMULATION", "")

        mol_weight, rot_bonds, globularity = None, None, None

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES at line {i+1}: {smiles}")
            else:
                mol = Chem.AddHs(mol)
                if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                    AllChem.UFFOptimizeMolecule(mol)

                    conf = mol.GetConformer()
                    coords = np.array([list(conf.GetAtomPosition(j)) for j in range(mol.GetNumAtoms())])
                    globularity = calculate_globularity(coords)

                    mol_weight = Descriptors.MolWt(mol)
                    rot_bonds = Descriptors.NumRotatableBonds(mol)
                else:
                    print(f"Embedding failed at line {i+1}: {smiles}")

        except Exception as e:
            print(f"Error processing line {i+1}: {e}")

        writer.writerow([smiles, accumulation, 
                         f"{mol_weight:.2f}" if mol_weight else "", 
                         rot_bonds if rot_bonds is not None else "", 
                         f"{globularity:.3f}" if globularity else ""])

print(f"Results saved to {output_file}")

