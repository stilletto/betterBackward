import time
import random
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from transformers.trainer_pt_utils import LabelSmoother
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForCausalLM


def generate_response(prompt, model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.batch_decode(generated_ids)
    return decoded_output[0].replace(prompt, "")

# print("generate_response")
# a = generate_response("### Instruction:\nUse the provided input to create an instruction that could have been used to generate the response with an LLM.\n\n### Input:\nI think it depends a little on the individual, but there are a number of steps you’ll need to take.  First, you’ll need to get a college education.  This might include a four-year undergraduate degree and a four-year doctorate program.  You’ll also need to complete a residency program.  Once you have your education, you’ll need to be licensed.  And finally, you’ll need to establish a practice.\n\n### Response:", model)
# print(a)



MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("code_llama", "LlamaForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("dbrx", "DbrxForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("git", "GitForCausalLM"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("mega", "MegaForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("open-llama", "OpenLlamaForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
    ]
)


def is_valid_smiles(smiles: str) -> bool:
    """
    Проверяет, является ли строка допустимой формулой SMILES.
    :param smiles: Строка, содержащая формулу SMILES.
    :return: True, если строка допустимая формула SMILES, иначе False.
    """
    if not smiles or len(smiles) < 2:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False



def generate_random_smiles(smiles: str) -> str:
    """
    Генерирует случайный SMILES из заданной формулы SMILES.
    :param smiles: Строка, содержащая формулу SMILES.
    :return: Строка, содержащая случайную формулу SMILES.
    """
    if not smiles:
        raise ValueError("Входная строка SMILES не должна быть пустой")

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Недопустимая формула SMILES")

        # Перестановка атомов для получения случайного SMILES
        atom_indices = list(range(mol.GetNumAtoms()))
        random.shuffle(atom_indices)
        random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)

        return random_smiles

    except Exception as e:
        raise ValueError(f"Ошибка при обработке SMILES: {e}")


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



import re

def is_valid_smiles(smiles: str) -> bool:
    """
    Check if the given SMILES string is a valid, complete SMILES formula.
    """
    from rdkit import Chem
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def is_potential_smiles_start(smiles: str) -> bool:
    """
    Check if the given SMILES string could be the start of a valid SMILES formula.
    """
    # Valid SMILES characters and patterns
    valid_chars = 'BCNOЗPSIclnops*.0123456789-=[]()/\\#%@+'
    valid_chars = 'BCNOPSFClBrIHbcnops*.0123456789-=[]()/\\#%@+<<>>:$+'


            # Check if all characters are valid SMILES characters
    for char in smiles:
        if char not in valid_chars:
            print("Invalid character %s" % char)
            return False

    # Check for unmatched square brackets, but allow one unclosed opening bracket at the end
    if smiles.count('[') < smiles.count(']') + 1:
        return False
    if smiles.count('[') > smiles.count(']') + 1:
        return False

    # Check for invalid atom definitions (unclosed square brackets except for the last one)
    if re.search(r"\[.*[^\]]$", smiles):
        if smiles[-1] != '[':
            print("Invalid atom definition")
            return False

    # Check for invalid bonds
    if re.search(r"[^A-Za-z0-9\]\)][-=:$/\\#]", smiles):
        print("Invalid bond")
        return False

    # Ensure that there are no unmatched parentheses (ignore the last character)
    open_parens = smiles.count('(')
    close_parens = smiles.count(')')
    if open_parens < close_parens:
        return False

    # Ensure that ring closure numbers are valid (i.e., pairs of numbers)
    ring_numbers = re.findall(r"\d", smiles)
    if len(ring_numbers) % 2 != 0:
        return False

    # Allow if the last character is a bond or open bracket (incomplete but valid start)
    if smiles[-1] in "-=#$:/\\[":
        return True

    # Ensure the last character isn't an invalid dangling bond or unclosed structure
    if smiles[-1] not in valid_chars:
        return False

    return True

# Example usage:
smiles_str = "Cc1[nH]c2nc(N)nc(O)c2c1Sc1cccc"
print(is_valid_smiles(smiles_str))  # Should return False for incomplete SMILES
print(is_potential_smiles_start(smiles_str))  # Should return True if it could be a valid start

# Additional test cases
smiles_valid = "CC(=O)O"
smiles_partial = "CC(=O"
print(is_valid_smiles(smiles_valid))  # Should return True for valid SMILES
print(is_potential_smiles_start(smiles_valid))  # Should return True for valid SMILES

print(is_valid_smiles(smiles_partial))  # Should return False for incomplete SMILES
print(is_potential_smiles_start(smiles_partial))  # Should return True if it could be a valid start    # unittest.main()