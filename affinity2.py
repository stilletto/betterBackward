import time

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

instruct_tune_dataset = load_dataset("mosaicml/instruct-v3")

instruct_tune_dataset = instruct_tune_dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")

instruct_tune_dataset["train"] = instruct_tune_dataset["train"].select(range(5_000))
instruct_tune_dataset["test"] = instruct_tune_dataset["test"].select(range(200))

def create_prompt(sample):
  bos_token = "<s>"
  original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
  response = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
  input = sample["response"]
  eos_token = "</s>"

  full_prompt = ""
  full_prompt += bos_token
  full_prompt += "### Instruction:"
  full_prompt += "\n" + system_message
  full_prompt += "\n\n### Input:"
  full_prompt += "\n" + input
  full_prompt += "\n\n### Response:"
  full_prompt += "\n" + response
  full_prompt += eos_token

  return full_prompt

create_prompt(instruct_tune_dataset["train"][0])



from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    device_map='auto',
    quantization_config=nf4_config,
    use_cache=False
)

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def generate_response(prompt, model):
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

import random
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



class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, model_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем model_teacher как атрибут экземпляра класса
        self.model_teacher = model_teacher
        self.label_smoother = LabelSmoother()
        self.save_outputs = []

    def compute_loss(self, model, inputs, return_outputs=False):
        #return self.compute_loss_random_peptide(model, inputs, return_outputs)
        """
        Compute the loss by separating input_ids and labels
        """

        # Get input_ids and labels from inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        labels = inputs.pop("labels")

        # # Decode the input_ids and labels to verify their content
        decoded_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        decoded_labels = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]

        # Debug: Print out the decoded inputs and labels to check their correctness
        for i in range(len(decoded_inputs)):
            print(f"Input {i}: {decoded_inputs[i]}")
            print(f"Label {i}: {decoded_labels[i]}")

        time.sleep(1000)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        # Compute the loss
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    def compute_loss_random_peptide(self, model, inputs, return_outputs=False):
        print("COMPUTE LOSS - RANDOM PEPTIDE LOSS")

        """
            How the loss is computed by Trainer. By default, all models return the loss in the first element.

            Subclass and override for custom behavior.
            """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            # If labels are provided as a list of multiple correct answers
            if isinstance(labels, list):
                losses = []
                for label in labels:
                    if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                        loss = self.label_smoother(outputs, label, shift_labels=True)
                    else:
                        loss = self.label_smoother(outputs, label)
                    losses.append(loss)
                loss = min(losses)
            else:
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
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss







from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)



args = TrainingArguments(
  output_dir = "mistral_instruct_generation",
  #num_train_epochs=5,
  max_steps = 100, # comment out this line if you want to train in epochs
  per_device_train_batch_size = 4,
  warmup_steps = 0.03,
  logging_steps=10,
  save_strategy="epoch",
  #evaluation_strategy="epoch",
  evaluation_strategy="steps",
  eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
  learning_rate=2e-4,
  bf16=True,
  lr_scheduler_type='constant',
)




max_seq_length = 2048

trainer = CustomSFTTrainer(
  model=model,
  peft_config=peft_config,
  max_seq_length=max_seq_length,
  tokenizer=tokenizer,
  packing=True,
  formatting_func=create_prompt,
  args=args,
  train_dataset=instruct_tune_dataset["train"],
  eval_dataset=instruct_tune_dataset["test"]
)

trainer.train()
trainer.save_model("peptide_test")



