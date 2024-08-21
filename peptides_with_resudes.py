import pandas as pd
from rdkit import Chem
from collections import Counter

# Чтение файла базы данных
file_name = "BioLiP_part_ALL_w_missing_SMILES_noh_mod_V2.txt"
df = pd.read_csv(file_name, sep="\t")

# Фильтрация необходимых колонок для генерации SMILES и активных остатков
columns = ['PDB_ID', 'PROTEIN_CHAIN', 'Prot_cont_resid_1', 'Receptor_seq', 'Peptide_SMILES']
data = df[columns]


# Функция для проверки валидности SMILES
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


# Функция для разметки активных остатков в последовательности белка
def mark_active_residues(receptor_seq, active_residues):
    residue_counter = Counter(active_residues)
    marked_seq = []

    for i, residue in enumerate(receptor_seq):
        residue_key = i + 1  # Нумерация последовательности начинается с 1
        if residue_key in active_residues:
            count = residue_counter[residue_key]
            if count == 1:
                marked_seq.append(f"({residue})")  # Один раз
            elif count == 2:
                marked_seq.append(f"[{residue}]")  # Два раза
            else:
                marked_seq.append(f"{{{residue}}}")  # Три и более раз
        else:
            marked_seq.append(residue)

    return ''.join(marked_seq)


# Группируем данные по PDB_ID и PROTEIN_CHAIN
grouped = data.groupby(['PDB_ID', 'PROTEIN_CHAIN'])

# Список для хранения данных
rows = []

# Генерация запросов для каждого таргета
for (pdb_id, chain), group in grouped:
    receptor_seq = group.iloc[0]['Receptor_seq']  # Последовательность белка одинаковая для всех строк этого таргета
    all_residues = []
    all_smiles = []

    # Проходим по строкам, собираем все активные остатки и SMILES-лиганды
    for index, row in group.iterrows():
        residues = row['Prot_cont_resid_1']
        if pd.notna(residues):
            # Преобразуем строку остатков в список чисел (например, "105,106,114" -> [105, 106, 114])
            all_residues.extend([int(res.strip()) for res in residues.split(',')])
        ligands = row['Peptide_SMILES']
        if pd.notna(ligands):
            smiles_list = ligands.split(',')
            valid_smiles = [smiles for smiles in smiles_list if is_valid_smiles(smiles)]
            all_smiles.extend(valid_smiles)

    # Убираем дубликаты из SMILES-лигандов
    all_smiles = list(set(all_smiles))

    # Обрабатываем последовательность белка, выделяя активные остатки
    marked_sequence = mark_active_residues(receptor_seq, all_residues)

    # Пояснение на английском
    explanation = ("Active residues are marked as follows: "
                   "single occurrence with ( ), double occurrence with [ ], "
                   "and more than two occurrences with { }.")

    # Формируем текст для новой строки
    if all_smiles:
        ligands_str = ','.join(all_smiles)
        text = (f"[INST] Generate new ligand in SMILES that have affinity to this target {marked_sequence} "
                f" known active residues marked as explained below. {explanation} [/INST]{ligands_str}")

        # Добавляем в список
        rows.append({"text": text})
    else:
        print(f"No valid SMILES found for PDB_ID: {pdb_id}, chain: {chain}. Skipping...")

# Создаем DataFrame из списка
output_df = pd.DataFrame(rows)

# Сохраняем DataFrame в CSV файл
output_df.to_csv('output_dataset.csv', index=False)

print("Dataset saved to 'output_dataset.csv'")
