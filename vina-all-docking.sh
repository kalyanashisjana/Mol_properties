for i in $(seq 1 368); do
obabel mol_${i}.pdb -O mol_${i}.pdbqt
./vina_1.2.5_linux_x86_64 --ligand mol_${i}.pdbqt --exhaustiveness 32 --config config.txt --out out-mol_${i}.pdbqt > mol_${i}.txt
done
