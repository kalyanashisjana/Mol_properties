import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D

# === Font configuration ===
font_directory = '/home/kalyan/python-font'
font_paths = fm.findSystemFonts(fontpaths=font_directory)
for font in font_paths:
    fm.fontManager.addfont(font)
rcParams['font.family'] = 'Calibri'

# === List your input files ===
files = [
    "molecules_set1.csv",
    "molecules_set2.csv",
    "molecules_set3.csv",
    "molecules_set4.csv",
    "molecules_set5.csv",
    "molecules_set6.csv"
]

# === Define colors and labels for each file ===
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
labels = ['ML 1', 'ML 2', 'ML 3', 'ML 4', 'ML 5', 'ML 6']

# === Create a 3D figure ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for i, filename in enumerate(files):
    try:
        data = pd.read_csv(filename)
        ax.scatter(
            data["Globularity"],
            data["Rotatable_Bonds"],
            data["Molecular_Weight"],
            color=colors[i % len(colors)],
            marker='o',
            label=labels[i],
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            s=60
        )
    except Exception as e:
        print(f"⚠️ Skipping {filename}: {e}")

# === Customize the plot ===
ax.set_title("Globularity vs Rotatable Bonds vs Molecular Weight", fontsize=24, fontweight="bold", pad=20)
ax.set_xlabel("Globularity", fontsize=18, fontweight="bold", labelpad=12)
ax.set_ylabel("Rotatable Bonds", fontsize=18, fontweight="bold", labelpad=12)
ax.set_zlabel("Molecular Weight (Da)", fontsize=18, fontweight="bold", labelpad=12)

ax.set_xlim(0.0, 0.6)
ax.set_ylim(0, 50)
ax.set_zlim(0, 1000)

ax.tick_params(axis='both', which='major', labelsize=14, width=1.5)
ax.legend(loc="upper left", fontsize=14, frameon=False)
ax.grid(alpha=0.3)
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig("globularity_rotbonds_molweight_all_sets.png", dpi=300, bbox_inches="tight")
plt.show()

