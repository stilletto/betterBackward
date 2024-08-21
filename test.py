import random
import time
import threading
import keyboard  # библиотека для отслеживания нажатий клавиш
import os
import sys

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForCausalLM
import torch
import numpy as np
from torch import nn
from trl import SFTTrainer
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers.trainer_pt_utils import LabelSmoother
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Dict, List, Union
import torch
from transformers import DataCollatorForLanguageModeling
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import sentencepiece
import bitsandbytes
import wandb
# Имя файла-сигнала
signal_file = 'stop_signal.txt'

def monitor_keyboard_signal(signal_file):
    while True:
        # Ждем нажатия комбинации клавиш Ctrl+S
        if keyboard.is_pressed('ctrl+s'):
            # Создаем файл-сигнал
            with open(signal_file, 'w') as f:
                f.write('stop')
            break
        time.sleep(0.1)




# Основной код обучения
# Загрузка датасета
dataset_name = "stilletto/target_affinity"



custom_loss = True
lora = True
#model_name = "models/mistral"
base_model = "models/mistral-inst"
base_model = "models/Mistral-7B-finetuned"
base_model = "models/Mistral-chem-finetuned7"
inst_model = True
#base_model = "models/Mistral-7B-finetuned"
model_name = base_model
device = "cuda"

separator0 = "[INST]"
separator1 = "[/INST]"
if inst_model:
    separator0 = "Create"
    separator1 = "format."


def filter_dataset(dataset_name, cache_dir=None):
    dataset = load_dataset(dataset_name, split="train", cache_dir=cache_dir)
    def filter_func(example):
        if separator1 in example['text']:
            index = example['text'].index('[/INST]')
            return index <= 800
        return False
    filtered_dataset = dataset.filter(filter_func)
    return filtered_dataset

dataset = filter_dataset(dataset_name)
print(f"dataset len={len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def merge_model(base_path, adapter_path, save_to):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_path)

    # Add/set tokens (same 5 lines of code we used before training)
    tokenizer.pad_token = "</s>"
    tokenizer.add_tokens(["<|im_start|>"])
    tokenizer.add_special_tokens(dict(eos_token="<|im_end|>"))
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.eos_token_id = tokenizer.eos_token_id

    # Load LoRA adapter and merge
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    model.save_pretrained(save_to, safe_serialization=True, max_shard_size='22GB')
    tokenizer.save_pretrained(save_to)


# merge_model(base_model, "checkpoint-4050", "models/Mistral-chem-finetuned7")
# print("merged")
# time.sleep(1000)


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm: bool = False):
        super().__init__(tokenizer, mlm=mlm)
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        texts = [feature['text'] for feature in features]
        alt_texts = []
        for i in range(2, 5):
            key = f'text{i}'
            alt_texts.append([feature.get(key, "") for feature in features])
        all_texts = texts + [item for sublist in alt_texts for item in sublist if item]
        batch = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
        return batch

def is_valid_smiles(smiles: str) -> bool:
    if not smiles or len(smiles) < 2:
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print("VALID SMILES!!!")
            return True
        else:
            return False
    except:
        return False

def find_amino_acids(mol):
    amino_acids_smarts = [
        "N[C@@H](C(=O)O)C", "N[C@@H](C(=O)O)C(C)C", "N[C@@H](C(=O)O)CC(C)C",
        "N[C@@H](C(=O)O)CC(=O)N", "N[C@@H](C(=O)O)CCC(=O)O", "N[C@@H](C(=O)O)CC(C)CC",
        "N[C@@H](C(=O)O)CS", "N[C@@H](C(=O)O)CC(C)O", "N[C@@H](C(=O)O)CCC(N)=O",
        "N[C@@H](C(=O)O)CC1=CNC=N1", "N[C@@H](C(=O)O)CCC1=CC=CC=C1",
        "N[C@@H](C(=O)O)CC1=CC=CC=N1", "N[C@@H](C(=O)O)CC(=O)NCC(=O)O",
        "N[C@@H](C(=O)O)[C@@H](O)C", "N[C@@H](C(=O)O)[C@@H](S)C"
    ]
    amino_acids_mols = [Chem.MolFromSmarts(aa) for aa in amino_acids_smarts]
    found_amino_acids = []
    for aa in amino_acids_mols:
        if mol.HasSubstructMatch(aa):
            found_amino_acids.append(aa)
    if len(found_amino_acids)>0:
        print("found amino acids!")
    return found_amino_acids

def has_peptide_bonds(mol):
    peptide_bond_smarts = [
        "C(=O)N", "C(=O)S", "C(=O)O"
    ]
    for smarts in peptide_bond_smarts:
        bond_mol = Chem.MolFromSmarts(smarts)
        if mol.HasSubstructMatch(bond_mol):
            print("HAS PEPTIDE BONDS!")
            return True
    return False

def has_cyclic_structure(mol):
    return mol.GetRingInfo().NumRings() > 0

def is_peptide(smiles: str) -> bool:
    if not is_valid_smiles(smiles):
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    amino_acids = find_amino_acids(mol)
    if not amino_acids:
        return False
    has_peptide_bonds_flag = has_peptide_bonds(mol)
    is_cyclic = has_cyclic_structure(mol)
    if is_cyclic and amino_acids:
        if has_peptide_bonds_flag:
            print("VALID PEPTIDE!!!!!!!!!!!!!!!!!!!!")
            return True
        else:
            return False
    n_term_smarts = "[N;H2]"
    c_term_smarts = "C(=O)[OH]"
    n_term_mol = Chem.MolFromSmarts(n_term_smarts)
    c_term_mol = Chem.MolFromSmarts(c_term_smarts)
    has_n_term = mol.HasSubstructMatch(n_term_mol)
    has_c_term = mol.HasSubstructMatch(c_term_mol)
    return has_peptide_bonds_flag and has_n_term and has_c_term

def loss_for_smiles(output):
    if not inst_model:
        if "/INST]" not in output:
            print("not even /INST]")
            return 10
        else:
            original_output = output
            output = output.split("/INST]")[1]
            if "</s>" in output:
                output = output.split("</s>")[0]
            if not is_valid_smiles(output):
                return 5
            if "peptide" in original_output.lower():
                if is_peptide(output):
                    return 0.1
            return 1
    else:
        if "format." not in output:
            print("not even format.")
            return 10
        else:
            original_output = output
            output = output.split("format.")[1]
            if "</s>" in output:
                output = output.split("</s>")[0]
            if not is_valid_smiles(output):
                return 5
            if "peptide" in original_output.lower():
                if is_peptide(output):
                    return 0.1
            return 1


def generate_random_smiles(smiles):
    # generate random SMILES from original canonical
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"wrong smiles={smiles}")
        print(is_valid_smiles(smiles))
        return None  # Возвращаем None, если SMILES некорректный

    # Генерируем случайную SMILES строку
    random_smiles = Chem.MolToSmiles(mol, doRandom=True)

    return random_smiles


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class advanced_SFTTrainer(SFTTrainer):
    def __init__(self, model, args, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None, label_smoother=None, **kwargs):
        super().__init__(
            model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset,
            tokenizer=tokenizer, model_init=model_init, **kwargs
        )
        self.label_smoother = LabelSmoother()
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get('input_ids')

        label = inputs.pop('labels')
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_labels = decoded_inputs
        decoded_inputs = [decoded_inputs[0].split(",")[0]]
        #print("Decoded inputs: ", decoded_inputs)
        pref = decoded_labels[0].split(",")[0]
        #print("Decoded labels: ", decoded_labels)
        pref = pref.split(separator1)[0] + separator1
        after_inst = decoded_labels[0].split(separator1)[1]
        label_list = after_inst.split(",")

        random_label_inputs = 0
        if len(label_list) > 1:
            random_label_inputs = random.randrange(0, len(label_list))
        labels = []
        max_length_input = len(decoded_inputs[0])
        for alt in label_list:
            a = pref + alt
            if len(a) > max_length_input:
                max_length_input = len(a)
        i = 0
        for _ in range(10):
            for alt in label_list:
                alt = generate_random_smiles(alt)
                if not alt:
                    continue
                a = pref + alt

                combo_label = [a]
                if i == random_label_inputs:
                    decoded_inputs = combo_label
                i += 1
                new_alt = self.tokenizer.batch_encode_plus(
                    combo_label, padding='max_length', max_length=max_length_input, truncation=False, return_tensors='pt'
                )['input_ids']
                labels.append(new_alt)
            if len(label_list)<=0:
                print("DATASET CORRUPTED!!!")
                time.sleep(99999)


        new_input_ids = self.tokenizer.batch_encode_plus(
            decoded_inputs,
            padding='max_length',
            truncation=False,
            max_length=max_length_input,
            return_tensors='pt'
        )['input_ids']
        new_attention_mask = torch.where(new_input_ids != 0, torch.tensor(1), torch.tensor(0)).to(model.device)
        inputs.update({'input_ids': new_input_ids, 'attention_mask': new_attention_mask})
        outputs = model(**inputs)
        logits = outputs.logits
        decoded_outputs = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        print("Decoded inputs: ", decoded_inputs)
        print("Decoded outputs: ", decoded_outputs)
        loss_multiplier = loss_for_smiles(decoded_outputs[0])
        print(f"Loss multiplies={loss_multiplier}")
        alternative_losses = []
        for alt_label in labels:
            alternative_loss = self.label_smoother(outputs, alt_label, shift_labels=True)
            alternative_losses.append(alternative_loss)
        min_loss = torch.min(torch.stack(alternative_losses)).to(model.device) * loss_multiplier
        print(f"minimal loss={min_loss}")
        return (min_loss, outputs) if return_outputs else min_loss


    def compute_loss_only_one_token(self, model, inputs, return_outputs=False):
        input_ids = inputs.get('input_ids')

        label = inputs.pop('labels')
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        decoded_labels = decoded_inputs
        decoded_inputs = [decoded_inputs[0].split(",")[0]]
        #print("Decoded inputs: ", decoded_inputs)
        pref = decoded_labels[0].split(",")[0]
        #print("Decoded labels: ", decoded_labels)
        pref = pref.split(separator1)[0] + separator1
        after_inst = decoded_labels[0].split(separator1)[1]
        label_list = after_inst.split(",")

        random_label_inputs = 0
        if len(label_list) > 1:
            random_label_inputs = random.randrange(0, len(label_list))
        labels = []
        max_length_input = len(decoded_inputs[0])
        for alt in label_list:
            a = alt
            if len(a) > max_length_input:
                max_length_input = len(a)
        i = 0
        for _ in range(10):
            for alt in label_list:
                alt = generate_random_smiles(alt)
                if not alt:
                    continue
                a = pref

                combo_label = [a]
                if i == random_label_inputs:
                    decoded_inputs = combo_label
                i += 1
                new_alt = self.tokenizer.batch_encode_plus(
                    combo_label, padding='max_length', max_length=max_length_input, truncation=False, return_tensors='pt'
                )['input_ids']
                labels.append(new_alt)
            if len(label_list)<=0:
                print("DATASET CORRUPTED!!!")
                time.sleep(99999)


        new_input_ids = self.tokenizer.batch_encode_plus(
            decoded_inputs,
            padding='max_length',
            truncation=False,
            max_length=max_length_input,
            return_tensors='pt'
        )['input_ids']
        #new_attention_mask = torch.where(new_input_ids != 0, torch.tensor(1), torch.tensor(0)).to(model.device)
        inputs.update({'input_ids': new_input_ids, 'labels': labels[0]})
        outputs = model(**inputs)
        logits = outputs.logits
        decoded_outputs = self.tokenizer.batch_decode(torch.argmax(logits, dim=-1), skip_special_tokens=True)
        print("Decoded inputs: ", decoded_inputs)
        print("Decoded outputs: ", decoded_outputs)
        loss_multiplier = loss_for_smiles(decoded_outputs[0])
        print(f"Loss multiplies={loss_multiplier}")
        alternative_losses = []
        for alt_label in labels:
            alternative_loss = self.label_smoother(outputs, alt_label, shift_labels=True)
            alternative_losses.append(alternative_loss)
        min_loss = torch.min(torch.stack(alternative_losses)).to(model.device) * loss_multiplier
        print(f"minimal loss={min_loss}")
        return (min_loss, outputs) if return_outputs else min_loss

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)
if device == "cpu":
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",
            trust_remote_code=True,
    )
else:
    if lora:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
        )
        #model.config.use_cache = False
        #model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=1024,
            bias="none",
            task_type="CAUSAL_LM",
            # use_rslora=True,
            # use_dora=True,
        )
        model = get_peft_model(model, peft_config)
    else:
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_config(model)

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            trust_remote_code=True,
        )

        # Загрузка контрольной точки и распределение между устройствами
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=base_model,  # Укажите путь к контрольной точке
            device_map="auto",  # Автоматическое распределение между GPU и CPU
            offload_folder="offload",  # Папка для выгрузки данных на диск
        )

        # Включение gradient checkpointing
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

LR = 0.0002
epochs = 100
max_length = 3000
expected_max_tokens = 1000

if device == "cpu":
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=15,
        logging_steps=3,
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
        use_cpu=True
    )

else:
    training_arguments = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=150,
        logging_steps=25,
        learning_rate=LR,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb",
        save_total_limit=3,
    )

if custom_loss:
    if device == "cpu":
        trainer = advanced_SFTTrainer(
            model=model,
            train_dataset=dataset,
            max_seq_length=1024,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )

    else:
        if lora:
            print("lora + cuda training")
            trainer = advanced_SFTTrainer(
                    model=model,
                    train_dataset=dataset,
                    peft_config=peft_config,
                    max_seq_length=1024,
                    dataset_text_field="text",
                    tokenizer=tokenizer,
                    args=training_arguments,
                    packing=False
                )
        else:
            trainer = advanced_SFTTrainer(
                model=model,
                train_dataset=dataset,
                max_seq_length=1024,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_arguments,
                packing=False
            )
else:
    if device == "cpu":
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            max_seq_length=1024,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )
    else:
        trainer = SFTTrainer(
                model=model,
                train_dataset=dataset,
                peft_config=peft_config,
                max_seq_length=1024,
                dataset_text_field="text",
                tokenizer=tokenizer,
                args=training_arguments,
                packing=False
            )



def handle_stop_signal(signal_file):
    global trainer
    try:
        os.remove(signal_file)  # Удаляем файл-сигнал
    except Exception:
        pass
    while True:
        if os.path.exists(signal_file):
            print("Получен сигнал остановки. Завершаем обучение...")
            trainer.control.should_training_stop = True
            trainer.save_model()  # Сохраняем модель
            new_model = "new_model"
            trainer.model.save_pretrained(new_model)
            base_path = base_model  # input: base model
            adapter_path = new_model  # input: adapters
            save_to = "models/Mistral-7B-finetuned"  # out: merged model ready for inference

            merge_model(base_path, adapter_path, save_to)

            model.config.use_cache = True
            try:
                os.remove(signal_file)  # Удаляем файл-сигнал
            except Exception:
                pass
            print("Saved, finish process")
            sys.exit(0)  # Завершаем процесс
        time.sleep(1)

# Запуск потока для отслеживания нажатий клавиш
keyboard_monitor_thread = threading.Thread(target=monitor_keyboard_signal, args=(signal_file,))
keyboard_monitor_thread.start()

# Запуск потока для обработки сигнала остановки
stop_signal_thread = threading.Thread(target=handle_stop_signal, args=(signal_file,))
stop_signal_thread.start()

trainer.train()

trainer.save_model()  # Сохраняем модель
new_model = "new_model"
trainer.model.save_pretrained(new_model)
base_path = base_model  # input: base model
adapter_path = new_model  # input: adapters
save_to = "models/Mistral-7B-finetuned2"  # out: merged model ready for inference

merge_model(base_path, adapter_path, save_to)

