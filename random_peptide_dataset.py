import random
from rdkit import Chem
from rdkit.Chem import AllChem


def is_valid_smiles(smiles: str) -> bool:
    """
    Проверяет, является ли строка допустимой формулой SMILES.
    :param smiles: Строка, содержащая формулу SMILES.
    :return: True, если строка допустимая формула SMILES, иначе False.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def find_amino_acids(mol):
    """
    Ищет и возвращает список найденных аминокислотных подструктур в молекуле.
    :param mol: Молекула RDKit.
    :return: Список подструктур аминокислот.
    """
    amino_acids_smarts = [
        "N[C@@H](C(=O)O)C", "N[C@@H](C(=O)O)C(C)C", "N[C@@H](C(=O)O)CC(C)C",
        "N[C@@H](C(=O)O)CC(=O)N", "N[C@@H](C(=O)O)CCC(=O)O", "N[C@@H](C(=O)O)CC(C)CC",
        "N[C@@H](C(=O)O)CS", "N[C@@H](C(=O)O)CC(C)O", "N[C@@H](C(=O)O)CCC(N)=O",
        "N[C@@H](C(=O)O)CC1=CNC=N1", "N[C@@H](C(=O)O)CCC1=CC=CC=C1",
        "N[C@@H](C(=O)O)CC1=CC=CC=N1", "N[C@@H](C(=O)O)CC(=O)NCC(=O)O",
        # Примеры модифицированных аминокислот
        "N[C@@H](C(=O)O)[C@@H](O)C", "N[C@@H](C(=O)O)[C@@H](S)C"
    ]

    amino_acids_mols = [Chem.MolFromSmarts(aa) for aa in amino_acids_smarts]
    found_amino_acids = []

    for aa in amino_acids_mols:
        if mol.HasSubstructMatch(aa):
            found_amino_acids.append(aa)

    return found_amino_acids


def has_peptide_bonds(mol):
    """
    Проверяет наличие различных типов пептидных связей в молекуле.
    :param mol: Молекула RDKit.
    :return: True, если найден хотя бы один тип пептидной связи, иначе False.
    """
    peptide_bond_smarts = [
        "C(=O)N",  # Амидные (пептидные) связи
        "C(=O)S",  # Тиоэфирные связи
        "C(=O)O"  # Изопептидные связи
    ]

    for smarts in peptide_bond_smarts:
        bond_mol = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(bond_mol):
            return True
    return False


def has_cyclic_structure(mol):
    """
    Проверяет наличие циклической структуры в молекуле.
    :param mol: Молекула RDKit.
    :return: True, если циклическая структура найдена, иначе False.
    """
    return mol.GetRingInfo().NumRings() > 0


def is_peptide(smiles: str) -> bool:
    """
    Проверяет, является ли формула SMILES пептидом.
    :param smiles: Строка, содержащая формулу SMILES.
    :return: True, если формула является пептидом, иначе False.
    """
    if not is_valid_smiles(smiles):
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    # Проверяем наличие аминокислотных подструктур
    amino_acids = find_amino_acids(mol)
    if not amino_acids:
        return False

    # Проверяем наличие различных типов пептидных связей
    has_peptide_bonds_flag = has_peptide_bonds(mol)

    # Проверяем наличие циклической структуры
    is_cyclic = has_cyclic_structure(mol)

    if is_cyclic and amino_acids:
        return has_peptide_bonds_flag  # Если молекула циклическая и содержит аминокислоты, и хотя бы один тип связи найден, считаем её пептидом

    # Дополнительно проверяем начало и конец цепочки (N-конец и C-конец) для линейных пептидов
    n_term_smarts = "[N;H2]"
    c_term_smarts = "C(=O)[OH]"

    n_term_mol = Chem.MolFromSmarts(n_term_smarts)
    c_term_mol = Chem.MolFromSmarts(c_term_smarts)

    has_n_term = mol.HasSubstructMatch(n_term_mol)
    has_c_term = mol.HasSubstructMatch(c_term_mol)

    return has_peptide_bonds_flag and has_n_term and has_c_term

def generate_random_peptide(length=5):
    amino_acids = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    peptide = ''.join(random.choices(amino_acids, k=length))
    mol = Chem.MolFromSequence(peptide)
    smiles = Chem.MolToSmiles(mol)
    if is_peptide(smiles):
        return smiles
    else:
        print("Invalid peptide, generating new one...")
        return generate_random_peptide(length)

def create_dataset(num_samples, output_file):
    dataset = set()
    instruction = "<s>[INST] Create new peptide in SMILES format. Just a random peptide in SMILES format, but please note that it must be new, that is, unknown to you. The peptide must be strictly one that was never present in the data on which you were trained.[/INST]"

    with open(output_file, 'w') as f:
        f.write("text\n")
        while len(dataset) < num_samples:
            smiles = generate_random_peptide()
            if smiles not in dataset:
                dataset.add(smiles)
                f.write(f'"{instruction}{smiles}</s>"\n')
            print("Samples generated:", len(dataset))

# Пример использования
num_samples = 1200000  # Количество образцов в датасете
output_file = 'dataset.csv'
create_dataset(num_samples, output_file)
print(f"Dataset with {num_samples} samples created and saved to {output_file}.")
