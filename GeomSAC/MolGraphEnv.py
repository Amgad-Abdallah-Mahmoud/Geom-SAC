import copy
from abc import ABC
from itertools import product

import gymnasium as gym
import numpy as np
import torch
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.utils import scatter

from utils import convert_radical_electrons_to_hydrogens


class MolecularGraphEnv(gym.Env, ABC):
    metadata = {"render.modes": ["human"]}

    def __init__(self, mol_g=None, max_atom=35, reward_type="qed", min_action=21, max_action=130,
                 allowed_atoms=["C", "Cl", "F", "I", "K", "N", "Na", "O", "S", "Br"], reference_mol=None,
                 target_sim=None, ):
        super(MolecularGraphEnv, self).__init__()
        self.mol_g = mol_g
        self.reward_type = reward_type
        self.min_action = min_action
        self.max_action = max_action
        self.counter = 0
        self.allowed_atoms = allowed_atoms
        self.max_atom = max_atom
        self.invalid_actions = 0

        self.reward_step = 0
        self.reward_type = reward_type
        self.reference_mol = reference_mol
        self.target_sim = target_sim

        self.possible_atom_types = np.array(allowed_atoms)
        self.possible_bond_types = np.array(
            [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC, ], dtype=object, )

        if type(self.mol_g) == str:
            self.mol_g = AllChem.MolFromSmiles(self.mol_g)
            self.total_bonds = len(self.mol_g.GetBonds())

        elif type(self.mol_g) == Chem.rdchem.Mol:
            self.mol_g = AllChem.RWMol(self.mol_g)
            self.total_bonds = len(self.mol_g.GetBonds())
        elif self.mol_g == None:
            self.total_bonds = 0
        else:
            raise ValueError

        self.action_space = gym.spaces.MultiDiscrete(
            [len(self.possible_atom_types), self.max_atom, self.max_atom, len(self.possible_bond_types), 4, ])
        self.stop = False

    def step(self, action):
        info = {}
        mol_old = copy.deepcopy(self.mol_g)
        mols = []

        stopping_cond = (self.counter >= self.max_action or self.invalid_actions > self.max_action)
        self.stop = True if stopping_cond else False

        if self.mol_g.GetNumAtoms() <= self.max_atom:
            self._add_atom(action[0])
        if ((self.mol_g.GetNumAtoms() > action[1]) and (self.mol_g.GetNumAtoms() > action[2]) and (
                action[1] != action[2])):
            node_id_1, node_id_2 = action[1], action[2]
        elif ((self.mol_g.GetNumAtoms() > action[1]) and (self.mol_g.GetNumAtoms() < action[2]) and (
                action[1] != action[2])):
            node_id_1, node_id_2 = action[1], self.mol_g.GetNumAtoms() - 1
            self.invalid_actions += 1
        elif ((self.mol_g.GetNumAtoms() < action[1]) and (self.mol_g.GetNumAtoms() > action[2]) and (
                action[1] != action[2])):
            node_id_1, node_id_2 = self.mol_g.GetNumAtoms() - 1, action[2]
            self.invalid_actions += 1
        else:
            node_id_1, node_id_2 = (self.mol_g.GetNumAtoms() - 2, self.mol_g.GetNumAtoms() - 1,)
            self.invalid_actions += 1
        try:
            if action[4] == 1:
                self._add_bond(action, node_id_1, node_id_2)
        except:
            self.invalid_actions += 1
            if len(mols) != 0:
                self.mol_g = mols[np.argmax([Chem.QED.qed(x) for x in mols])]
            else:
                self.mol_g = mol_old
        try:
            if action[4] == 0:
                self._connect_mol_frags()
        except:
            self.invalid_actions += 1
            if len(mols) != 0:
                self.mol_g = mols[np.argmax([Chem.QED.qed(x) for x in mols])]
            else:
                self.mol_g = mol_old
        try:
            if action[4] == 2:
                self._alter_bond(action, node_id_1, node_id_2)
        except:
            self.invalid_actions += 1
            if len(mols) != 0:
                self.mol_g = mols[np.argmax([Chem.QED.qed(x) for x in mols])]
            else:
                self.mol_g = mol_old
        try:
            if action[4] == 3:
                self._remove_bond(node_id_1, node_id_2)

        except:
            self.invalid_actions += 1
            if len(mols) != 0:
                self.mol_g = mols[np.argmax([Chem.QED.qed(x) for x in mols])]
            else:
                self.mol_g = mol_old
        mols.append(self.mol_g)

        if self.check_valency():
            self.reward_step += 1 / self.max_action
        else:
            self.reward_step -= 1 / self.max_action
            self.mol_g = mol_old
            self.invalid_actions += 1

        if self.check_chemical_validity():
            current_qed = np.sqrt(np.power(Chem.QED.qed(self.mol_g), -1))
            self.reward_step += (1.5 - current_qed) / 20

        if self.check_stereo():
            self.reward_step += 1 / self.max_action
        else:
            self.reward_step -= 1 / self.max_action
            self.mol_g = mol_old
            self.invalid_actions += 1
        try:
            if self.reference_mol is not None and self.target_sim is not None:
                self.reward_step += (1.5 - (np.sqrt(self.get_similarity()) ** -1)) / 20
        except:
            self.reward_step -= 1 / self.max_action

        if self.stop or self.counter >= self.max_action:
            reward_valid = 2
            reward_qed = 0
            reward_final = 0
            reward_geom = 2

            if not self.check_stereo():
                reward_geom -= 5
            try:
                if self.reference_mol is not None and self.target_sim is not None:
                    reward_final += (1.5 - (np.sqrt(self.get_similarity()) ** -1)) * 3
            except:
                reward_final -= 2

            if not self.check_chemical_validity():
                reward_valid -= 5
            else:
                # final mol object where any radical electrons are changed to bonds to hydrogen
                final_mol = self.get_final_mol()
                s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                final_mol = Chem.MolFromSmiles(s)

                try:
                    if self.check_stereo():
                        reward_geom += 2
                        reward_final += reward_geom
                    else:
                        reward_geom -= 1
                        reward_final += reward_geom

                    if self.check_valency():
                        reward_valid += 2
                        reward_final += reward_valid
                    else:
                        reward_valid -= 1
                        reward_final += reward_valid
                    reward_qed += Chem.QED.qed(final_mol)
                    if self.reward_type == "qed":
                        reward_final += reward_qed * 2
                    else:
                        print("reward error!")
                        reward_final = 0

                except:
                    print("reward error")

            new = True
            reward = self.reward_step + reward_valid + reward_final

            info["reward_valid"] = reward_valid
            info["reward_qed"] = reward_qed
            info["reward_structure"] = reward_geom
            info["final_stat"] = reward_final
            info["reward"] = reward
            info["stop"] = self.stop

        else:
            new = False
            reward = self.reward_step

        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0
            self.invalid_actions = 0
            self.reward_step = 0
        return ob, reward, new, info

    def reset(self, frame_work='pyg'):
        if self.mol_g is not None:
            self.mol_g = AllChem.RWMol(self.mol_g)

        if self.mol_g is None:
            self.mol_g = AllChem.RWMol()
            self._add_atom(0)

        self.counter = 0
        self.invalid_actions = 0
        if frame_work == 'pyg':
            ob = self.get_observation(frame_work='pyg')
        if frame_work == 'dgl':
            ob = self.get_observation(frame_work='dgl')
        return ob

    def _add_atom(self, atom_id):
        atom_symbol = self.possible_atom_types[atom_id]
        self.mol_g = Chem.RWMol(self.mol_g)
        self.mol_g.AddAtom(Chem.Atom(atom_symbol))

    def _add_bond(self, action, node_id_1, node_id_2):
        bond_type = self.possible_bond_types[action[3]]
        bond = self.mol_g.GetBondBetweenAtoms(int(node_id_1), int(node_id_2))
        if bond:
            return False
        else:
            self.mol_g.AddBond(int(node_id_1), int(node_id_2), order=bond_type)

    def _alter_bond(self, action, node_id_1, node_id_2):
        bond_type = self.possible_bond_types[action[3]]
        bond = self.mol_g.GetBondBetweenAtoms(int(node_id_1), int(node_id_2))
        if bond:
            bond.SetBondType(bond_type)
        else:
            self.mol_g.AddBond(int(node_id_1), int(node_id_2), order=bond_type)

    def _remove_bond(self, node_id_1, node_id_2):
        bond = self.mol_g.GetBondBetweenAtoms(int(node_id_1), int(node_id_2))
        if bond:
            self.mol_g.RemoveBond(int(node_id_1), int(node_id_2))
        else:
            return False

    def _connect_mol_frags(self):
        pt = Chem.GetPeriodicTable()
        valid_conn = []
        qed = []
        mols = []
        ps = []
        old_mol = copy.deepcopy(self.mol_g)
        frags = Chem.GetMolFrags(old_mol)
        Chem.SanitizeMol(old_mol)
        for a in old_mol.GetAtoms():
            if a.GetExplicitValence() < pt.GetDefaultValence(a.GetAtomicNum()):
                valid_conn.append(a.GetIdx())
        if len(valid_conn) > 1:
            for i in valid_conn:
                for f in frags:
                    if i in f:
                        ps.append(str(i))
                    else:
                        ps.append("s")
        else:
            pass
        ps = " ".join(ps).split(" s s ")
        ps = [x.replace("s", "").split() for x in ps]
        ps = list(product(*ps))
        if len(frags) < 1:
            for i in ps:
                node_1 = int(i[0])
                node_2 = int(i[1])
                for k in self.possible_bond_types:
                    try:
                        n = copy.deepcopy(old_mol)
                        n.AddBond(node_1, node_2, order=k)
                        m3 = Chem.AddHs(n, addCoords=True)
                        AllChem.EmbedMolecule(m3)
                        AllChem.MMFFOptimizeMolecule(m3, maxIters=2000)
                        qed.append(Chem.QED.qed(n))
                        mols.append(n)
                    except:
                        pass
        if len(qed) != 0:
            mol_idx = np.argmax(qed)
            mol = mols[mol_idx]
        elif len(frags) < 1:
            mol = get_final_mols(old_mol)
        else:
            mol = self.mol_g
        Chem.SanitizeMol(mol)
        return mol

    def get_num_atoms(self):
        return len(self.mol_g.GetNumAtoms())

    def get_num_bonds(self):
        return len(self.mol_g.GetNumBonds())

    def check_chemical_validity(self):
        try:
            s = Chem.MolToSmiles(self.mol_g, isomericSmiles=True)
            m = Chem.MolFromSmiles(s)
        except:
            return False

        if m is None:
            return False
        else:
            try:
                if Chem.SanitizeMol(m):
                    return True
                else:
                    return False
            except:
                return False

    def check_stereo(self):
        m = copy.deepcopy(self.mol_g)
        rw = Chem.RWMol(m)
        stereo_state = False
        try:
            Chem.SanitizeMol(rw)
            m3 = Chem.AddHs(rw)
            AllChem.EmbedMolecule(m3)
            if AllChem.MMFFOptimizeMolecule(m3) == 0:
                stereo_state = True
            else:
                stereo_state = False
        except:
            stereo_state = False
        return stereo_state

    def check_mol_conn(self):
        return len(Chem.GetMolFrags(self.get_final_mol()))

    def check_valency(self):
        try:
            Chem.SanitizeMol(self.mol_g, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
            return True
        except ValueError:
            return False

    def get_final_mol(self):
        m = convert_radical_electrons_to_hydrogens(self.mol_g)
        Chem.SanitizeMol(m)
        return m

    def get_similarity(self):
        n = copy.deepcopy(self.reference_mol)
        m = copy.deepcopy(self.mol_g)

        fp_n = AllChem.GetMorganFingerprint(n, radius=2)
        fp_m = AllChem.GetMorganFingerprint(m, radius=2)
        curr_sim = DataStructs.TanimotoSimilarity(fp_n, fp_m)

        return 1 - np.sqrt(abs(self.target_sim - curr_sim) / (self.target_sim + curr_sim) ** 2)

    def get_observation(self, frame_work='pyg'):
        mol_ob = copy.deepcopy(self.mol_g)

        atom_types = self.possible_atom_types.tolist()
        types = dict(zip(atom_types, [x for x in range(len(atom_types))]))
        bonds = dict(zip(self.possible_bond_types[:4], range(len(self.possible_bond_types[:4]))))

        N = mol_ob.GetNumAtoms()

        type_idx = []
        atomic_number = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        atom_degree = []

        for atom in mol_ob.GetAtoms():
            type_idx.append(types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == Chem.rdchem.HybridizationType.SP else 0)
            sp2.append(1 if hybridization == Chem.rdchem.HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == Chem.rdchem.HybridizationType.SP3 else 0)
            atom_degree.append(Chem.rdchem.Atom.GetDegree(atom))

        z = torch.tensor(atomic_number, dtype=torch.long)

        try:
            m3 = copy.deepcopy(mol_ob)
            m3 = Chem.AddHs(mol_ob, addCoords=True)
            AllChem.EmbedMolecule(m3)
            AllChem.MMFFOptimizeMolecule(m3)
            conf = m3.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)
        except:
            self.stop = True

        row, col, edge_type = [], [], []
        for bond in mol_ob.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            edge_type += 2 * [bonds[bond.GetBondType()]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_type = torch.tensor(edge_type, dtype=torch.long)
        edge_attr = one_hot(edge_type, num_classes=len(self.possible_bond_types))

        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N, reduce="sum").tolist()

        x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
        x2 = (torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs, atom_degree],
                           dtype=torch.float, ).t().contiguous())
        x = torch.cat([x1, x2], dim=-1)
        if frame_work == 'dgl':
            ob = dgl.graph((row, col))
            ob.ndata["x"] = x
            ob.edata["edge_attr"] = edge_attr
        if frame_work == 'pyg':
            ob = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return ob
