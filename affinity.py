import time
import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
except ImportError:
    os.system("pip install transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
try:
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
except ImportError:
    os.system("pip install peft")
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

try:
    import wandb
except ImportError:
    os.system("pip install wandb")
    import wandb
from datasets import load_dataset
import datasets

from datasets import Dataset, DatasetDict
try:
    from trl import SFTTrainer
except ImportError:
    os.system("pip install trl")
    from trl import SFTTrainer
try:
    from openai import OpenAI
except ImportError:
    os.system("pip install openai")
    from openai import OpenAI
from transformers.trainer_pt_utils import LabelSmoother
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
import subprocess
import sys
import concurrent.futures
import re
import unicodedata
import string
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils import seq1



custom_loss = True
device = "cuda"

wb_key = "f56a3ab3548dc1c8f416b4525d6af324963a15f8"
#wb_key = "local-5213b45cda9492f5c21fea9ac8407cac36cf3141"
#dataset_name = "mlabonne/guanaco-llama2-1k"
dataset_name = "stilletto/target_affinity"
dataset_name = "stilletto/random_peptide_small"
#dataset_name = "centroIA/MistralInstructScenariosv2"
new_model = "mistral_7b_affinity"

base_model = "models/mistral"
LR = 0.0002 #default
#LR = 0.008 #new
epochs = 1000
max_length = 3000  # Контекстное окно модели
expected_max_tokens = 1000  # Максимальное количество ожидаемых токенов после генерации

# tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
# tokenizer.padding_side = 'left'
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.add_eos_token = True
# tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True


def test_model(model4test):
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=False,
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model4test,
        #quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    prompt = "<s>[INST] What is your favourite condiment? [/INST]" \
             "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> " \
             "[INST] Do you have mayonnaise recipes? [/INST]"
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    #generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_ids = model.generate(**model_inputs, max_new_tokens=expected_max_tokens)
    decoded = tokenizer.batch_decode(generated_ids)
    print(decoded)


#model4test = "results/checkpoint-325"
model4test = base_model
# test_model(model4test)
# time.sleep(1000)


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

        random_smiles = Chem.MolToRandomSmiles(mol)
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




def GPT_trainer(input_text,predicted_text,ground_truth_text):
    #print("input_text= %s"%input_text[:10])
    print("predicted_text= %s"%predicted_text[:10])
    #print("ground_truth_text= %s"%ground_truth_text[:10])
    return ground_truth_text #todo temporary only for debug
    client = OpenAI(api_key="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")

    question = str({"input_text": input_text,
                    "predicted_text": predicted_text,
                    "ground_truth_text": ground_truth_text})

    rejected_text = "I'm sorry, I can't answer that question. Please ask me something else."
    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "Your task is to evaluate the text according to the following rules. You will have the ground_truth_text and the predicted_text. If the predicted_text is an adequate part of the ground_truth_text, simply repeat it. If the last word or part of a word of the predicted_text is incorrect, then change the text so that it is as similar as possible in the arrangement of words and their choice to the predicted_text but in meaning is similar to the ground_truth_text. That is, correct it, trying to minimally change the predicted_text. But if the predicted_text is correct, including the correct part, do not change it, do not answer anything unnecessary, just return it as is. Examples:\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Москв\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is correct predicted part of text. Because it is part of right answer \"Москва\" . You must to return \"Москв\"\n\nIncorrect:\n {\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Москву\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is incorrect predicted part of text. Because last latter is wrong. You must to return \"Москва\" because it is most close right answer\n\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Насколько я знаю Москва\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is correct predicted text or part of text. Because it is content right. The sentence is constructed differently, but the meaning is the same and it is correct . You must to return \"Насколько я знаю Москва\"\n\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Мо\",\n\"ground_truth_text\": \"Самый большой город России - Москва\",\n}\nIt is correct predicted part of text. Because it is part of right answer \"Москва\". You must to return \"Мо\"\n\nCorrect:\n{\"input_text\": \"Какой самый большой город Юпитера?\",\n\"predicted_text\": \"Не\",\n\"ground_truth_text\": \"На Юпитере нет городов.\"\n}\nIt is correct predicted part of text. Because further the phrase can be predicted as, for example, “I don’t know” or “There are no cities on Jupiter” and these will be the correct answers, so we consider this the correct beginning of the phrase.  You must to return \"Не\"\n\n\nIncorrect:\n{\"input_text\": \"Какой самый большой город Юпитера?\",\n\"predicted_text\": \"..i\",\n\"ground_truth_text\": \"На Юпитере нет городов.\"\n}\nIt is incorrect predicted part of text.  This is the wrong beginning of a phrase, no matter what the next phrase is, the correct answer can hardly begin like that. Therefore, return the same token in the account, which is the last in this case. Since the correct first one would be \"На\". You must to return \"На\"\n\nAdditional instructions:\nIf there are obvious errors in the predicted text, correct them in your answer. For example: \"predicted_text\" = \"The sea was compassionately yellow\" We can assume that the sea was yellow, for example in a fantasy work, if in the general sense it agrees with ground_truth_text, but it could not possibly be \"compassionately yellow\" Therefore, we need to return \"The sea was yellow\" or if it should have been blue based on ground_truth_text or if there is no reason to believe that it could have been yellow then the answer will be \"The sea was blue\"\nIf the predicted text is a response to a request to write some kind of free story, then return everything as is, correcting only spelling errors, factual errors and errors in the coordination of sentences, as well as errors when the style of the text clearly does not correspond to the request. When correcting them, try to keep all the text as much as possible in the same form as it was in predicted_text\n"
                },
                {
                    "role": "user",
                    "content": "{\"input_text\":" + question + " }"
                }
            ],
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        response = response.choices[0].message.content
        if rejected_text not in response:
            break
    try:
        if '"' == response[0]:
            response = response[1:]
        if '"' == response[-1]:
            response = response[:-1]
    except Exception:
        pass
    print(response)
    return str(response)


wandb.login(key=wb_key)
print("wandb itinializing")
wandb.init(
    # set the wandb project where this run will be logged
    project="bbw",
    # track hyperparameters and run metadata
    config={
    "learning_rate": LR,
    "architecture": "CNN",
    "dataset": dataset_name,
    "epochs": epochs,
    }
)
#wandb.init(project="bbw")


print("wandb initialized")
# time.sleep(100)




def preprocess_dataset(dataset_name, max_tokens, expected_max_tokens, model_name):
    print("preprocessing dataset")
    datasets.disable_caching()
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def is_english(text): # todo temporary only for debug
        printable = set(string.printable)
        a = all(char in printable for char in text)
        if a:
            print("english")
        else:
            print("not english")
        return a

    def filter_long_and_no_ins_and_english_only(example):
        print("-------")
        text = example['text']
        text_tokens = tokenizer.encode(text, return_tensors='pt')
        if text_tokens.size(-1) > (max_tokens - expected_max_tokens):
            print("too long")
            return False
        elif not re.search(r'\[INST\]', text):
            print("no instructions")
            return False
        elif not is_english(text): # todo temporary only for debug
            print("not english")
            return False
        return True

    dataset = dataset.filter(filter_long_and_no_ins_and_english_only)
    return dataset






# def custom_loss_with_teacher_approval_v2(input_text, model_output_logits, expected_tokens_indices, model_teacher, tokenizer):
#     """
#     :param model_output_logits: Логиты, полученные от модели-ученика.
#     :param expected_tokens_indices: Индексы ожидаемых токенов из датасета.
#     :param model_teacher: Экземпляр модели-учителя с методом approve_text, принимающим и возвращающим текст.
#     :param tokenizer: Токенизатор, связанный с моделью-учеником.
#     """
#
#     # Получаем наиболее вероятные индексы предсказанных токенов
#     predicted_indices = torch.argmax(F.softmax(model_output_logits, dim=-1), dim=-1)
#
#     # Конвертируем предсказанные индексы и истинные индексы в текст
#     #predicted_text = tokenizer.decode(predicted_indices.tolist(), skip_special_tokens=True)
#     predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_indices.tolist()]
#     predicted_text = predicted_texts[0]
#     expected_tokens_indices = [[max(idx, 0) for idx in sequence] for sequence in expected_tokens_indices]
#     # if expected_tokens_indices is not None:
#     expected_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in
#                       expected_tokens_indices]
#     # else:
#     #     expected_texts = [None] * len(predicted_texts)  # Если нет лейблов
#
#     # Модель-учитель оценивает предсказанный текст и возвращает "одобренный" текст
#     expected_text = expected_texts[0]
#     #predicted_text2 = predicted_text[len(input_text):]
#     #expected_text = expected_text[len(input_text):]
#     # print("input_text= %s" % input_text)
#     # print("=======================================")
#     # print("predicted_text= %s" % predicted_text)
#     # print("---------------------------------------")
#
#
#     last_inst_index = expected_text.rfind("[/INST]")  # Индекс последнего вхождения
#     if last_inst_index != -1:
#         expected_text = expected_text[last_inst_index + len("[/INST]"):]  # Получаем текст до последнего включительно
#     else:
#         #raise ValueError("No [INST] tag found in expected_text. Dataset not clear or not for instruct model")
#         pass
#     #print("expected_text= %s" % expected_text)
#     #time.sleep(3000)
#
#     # if "[INST]" in expected_text:
#     #     expected_text = expected_text.split("[INST]")[1]
#     # if "[/INST]" in expected_text:
#     #     expected_text = expected_text.split("[/INST]")[1]
#     # if "[INST]" in input_text:
#     #     input_text = input_text.split("[INST]")[1]
#     # if "[/INST]" in input_text:
#     #     input_text = input_text.split("[/INST]")[0]
#     # last_inst_index = input_text.rfind("[/INST]")  # Индекс последнего вхождения
#     # if last_inst_index != -1:
#     #     input_text = input_text[:last_inst_index + len("[/INST]")]  # Получаем текст до последнего включительно
#     # else:
#     #     raise ValueError("No [INST] tag found in input_text. Dataset not clear or not for instruct model")
#     # expected_text = expected_text.replace("<s>", "")
#     # expected_text = expected_text.replace("</s>", "")
#     if not custom_loss:
#         return expected_text # Если не используется кастомная функция потерь, возвращаем ожидаемый текст но вообще без разницы что возвращать
#     approved_text = GPT_trainer(input_text, predicted_text, expected_text)
#
#     #approved_text = approve_text(model_teacher, predicted_text, expected_text)
#     #print("approved_text= %s" % approved_text[:50])
#
#     # Конвертируем "одобренный" текст назад в индексы токенов
#     approved_tokens_ids = torch.tensor(tokenizer.encode(approved_text, add_special_tokens=True), dtype=torch.long)
#
#     # # Расчет функции потерь; примечание - возможно потребуется настройка размерностей
#     # #loss = F.cross_entropy(model_output_logits[:, :approved_tokens_ids.size(0)], approved_tokens_ids.unsqueeze(0))
#     # print("model_output_logits shape")
#     # print(model_output_logits.shape)
#     # print("approved_tokens_ids shape")
#     # print(approved_tokens_ids.shape)
#     # loss = trainer.label_smoother(model_output_logits, approved_tokens_ids, shift_labels=True)
#     # #loss = F.cross_entropy(model_output_logits, approved_tokens_ids)
#
#     return approved_tokens_ids

from collections import OrderedDict


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


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, model_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем model_teacher как атрибут экземпляра класса
        self.model_teacher = model_teacher
        self.label_smoother = LabelSmoother()
        self.save_outputs = []

    def decode_labels(self, labels, tokenizer):
        # Заменяем все -100 на tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == -100] = tokenizer.pad_token_id

        # Декодируем метки
        decoded_labels = []
        for label in labels:
            decoded_label = tokenizer.decode(label, skip_special_tokens=True)
            decoded_labels.append(decoded_label)
        return decoded_labels

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """


        if self.label_smoother is not None and "labels" in inputs:
            # Извлечение меток из входных данных, если они присутствуют
            labels = inputs.pop("labels")
        else:
            labels = None

        decoded_labels = self.decode_labels(labels, tokenizer)

        print(decoded_labels)
        time.sleep(1000)

        # labels = labels.clone()
        # labels[labels == -100] = tokenizer.pad_token_id

        # # Декодируем метки
        # decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

        i = 0
        print(len(labels))
        for label in labels:
            a = tokenizer.decode(label)
            print(f"label {i}={a}")

        time.sleep(1000)

        # Передача оставшихся входных данных в модель для получения предсказаний
        outputs = model(**inputs)
        decoded = tokenizer.batch_decode(outputs.logits.argmax(dim=-1))
        print(decoded)

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # Если метки присутствуют, вычисление потерь с использованием меток и предсказаний модели
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # Использование потерь из предсказаний модели
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     return self.compute_loss_random_peptide(model, inputs, return_outputs)
    #     print("COMPUTE LOSS - SMART LOSS")
    #     """
    #     How the loss is computed by Trainer. By default, all models return the loss in the first element.
    #
    #     Subclass and override for custom behavior.
    #     """
    #     if self.label_smoother is not None and "labels" in inputs:
    #         labels = inputs.pop("labels")
    #     else:
    #         labels = None
    #         raise ValueError("NO LABELS!")
    #
    #     labels = [[max(idx, 0) for idx in sequence] for sequence in labels]
    #     # if expected_tokens_indices is not None:
    #     expected_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in
    #                       labels]
    #
    #     print("expected_texts= %s" % expected_texts)
    #     time.sleep(1000)
    #
    #
    #     outputs = model(**inputs)
    #
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]
    #
    #     if labels is not None:
    #         unwrapped_model = unwrap_model(model)
    #
    #         if _is_peft_model(unwrapped_model):
    #             model_name = unwrapped_model.base_model.model._get_name()
    #         else:
    #             model_name = unwrapped_model._get_name()
    #
    #         if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    #             shift_labels = True
    #         else:
    #             shift_labels = False
    #
    #         loss_fct = self.label_smoother
    #         loss_values = []
    #         for key in labels.keys():
    #             loss_values.append(loss_fct(outputs.logits[:, int(key[-1])], labels[key], shift_labels=shift_labels))
    #
    #         loss = min(loss_values)
    #     else:
    #         raise ValueError(
    #             "NO LABELS!"
    #         )
    #
    #     return (loss, outputs) if return_outputs else loss


#ЗАПУСТИТЬ ПЕРЕД ВЫБОРОМ НОВОГО ДАТАСЕТА ДЛЯ ОЧИСТКИ!!!
# dataset = preprocess_dataset(dataset_name, max_length, expected_max_tokens, base_model)
# new_dataset_name = dataset_name.split("/")[-1] + "_processed"
# repo_name = "stilletto/" + new_dataset_name
# print(repo_name)
# dataset.push_to_hub(repo_name)
# #Importing the dataset
dataset = load_dataset(dataset_name, split="train")
dataset.cleanup_cache_files()



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
)


def format_text_for_sft_trainer(text):
    # Удаление начального и конечного тегов <s> и </s>
    text = re.sub(r'^<s>', '', text)
    text = re.sub(r'</s>$', '', text)

    # Удаление тегов [INST] и [/INST]
    text = re.sub(r'\[INST\]', '', text)
    text = re.sub(r'\[/INST\]', '', text)

    # Дополнительные шаги для форматирования текста можно добавить здесь

    return text.strip()


# Пример использования:
example_text = "<s>[INST] Create new peptide in SMILES format. Just a random peptide in SMILES format, but please note that it must be new, that is, unknown to you. The peptide must be strictly one that was never present in the data on which you were trained.[/INST]NC(=O)CC[C@H](NC(=O)[C@@H](N)Cc1ccccc1)C(=O)N[C@@H](Cc1ccccc1)C(=O)NCC(=O)N[C@@H](CCC(=O)O)C(=O)O</s>"
formatted_text = format_text_for_sft_trainer(example_text)
print(formatted_text)


model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()



model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)

model = get_peft_model(model, peft_config)

training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=epochs,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=LR,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="wandb",
    save_total_limit=3,
)

# trainer = CustomSFTTrainer(
#         model=model,
#         train_dataset=dataset,
#         peft_config=peft_config,
#         max_seq_length=max_length,
#         tokenizer=tokenizer,
#         args=training_arguments,
#         dataset_text_field="text",
#         packing=False)


trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_length,
        tokenizer=tokenizer,
        args=training_arguments,
        dataset_text_field="text",
        packing=False)


trainer.train()
trainer.model.save_pretrained(new_model)
print("training finished")
# wandb.finish()
model.config.use_cache = False

model4test = "results/checkpoint-339"
test_model(model4test)

#
#
#
# model_path = "C:\\LocalRepo\\betterBackward\\models\\Mistral-7B-v0.1"
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, device_map="auto", load_in_4bit=True
# )
# # model_path = "models/Mistral-7B-v0.1"
# # model_path = "C:\\LocalRepo\\betterBackward\\models\\Mistral-7B-v0.1"
# # tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
# # model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")
#
#
# # generated_ids = model.generate(**model_inputs)
# # tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# # tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
# # model_inputs = tokenizer(
# #     ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
# # ).to("cuda")
# # generated_ids = model.generate(**model_inputs)
# # output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# #print(output)
# print("done----------")
# if __name__ == '__main__':
#     pass
#
