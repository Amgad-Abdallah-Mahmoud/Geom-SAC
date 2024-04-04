def convert_radical_electrons_to_hydrogens(mol):
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m

def get_final_mols(m):
    mols = list(Chem.rdmolops.GetMolFrags(m, asMols=True))
    if mols:
        mols.sort(reverse=True, key=lambda m: m.GetNumAtoms())
        mol = mols[0]
        return mol
