import time

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, AutoModelForCausalLM
import torch
import numpy as np
from torch import nn
from trl import SFTTrainer

# Загрузка датасета
dataset_name = "stilletto/target_affinity"
#dataset = load_dataset('json', data_files='output_dataset.json')
dataset = load_dataset(dataset_name, split="train", cache_dir=None)
print(dataset)
# Токенизатор
model_name = "models/mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Добавление pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# def parse_string_to_data(examples):
#     input_ids_batch = []
#     attention_mask_batch = []
#     labels_batch = []
#     alternative_labels_batch = []
#
#     for example in examples['text']:
#         # Удаляем теги <s> и </s>
#         example_string = example.strip('<s>').strip('</s>')
#         # Разделяем по тегу [INST] и [/INST]
#         input_part, outputs_part = example_string.split('[/INST]')
#         input_part = input_part.strip('[INST]').strip()
#         # Разделяем outputs по запятым
#         outputs = [output.strip() for output in outputs_part.split(',')]
#
#         # Токенизируем input и outputs
#         input_ids = tokenizer.encode(input_part, add_special_tokens=False)
#         attention_mask = [1] * len(input_ids)
#         labels = tokenizer.encode(outputs[0], add_special_tokens=False)
#         if len(outputs) > 1:
#             alternative_labels = [tokenizer.encode(output, add_special_tokens=False) for output in outputs[1:]]
#         else:
#             alternative_labels = []
#
#         input_ids_batch.append(input_ids)
#         attention_mask_batch.append(attention_mask)
#         labels_batch.append(labels)
#         alternative_labels_batch.append(alternative_labels if alternative_labels else [[-100] * len(labels)])
#
#     max_len = max(max(len(ids) for ids in input_ids_batch),
#                   max(len(lbls) for lbls in labels_batch),
#                   max(max(len(alt) for alt in alts) for alts in alternative_labels_batch))
#
#     for idx in range(len(input_ids_batch)):
#         input_ids_batch[idx] += [tokenizer.pad_token_id] * (max_len - len(input_ids_batch[idx]))
#         attention_mask_batch[idx] += [0] * (max_len - len(attention_mask_batch[idx]))
#         labels_batch[idx] += [-100] * (max_len - len(labels_batch[idx]))
#         alternative_labels_padded = []
#         for alt in alternative_labels_batch[idx]:
#             alternative_labels_padded.append(alt + [-100] * (max_len - len(alt)))
#         alternative_labels_batch[idx] = alternative_labels_padded
#
#     return {
#         'input_ids': input_ids_batch,
#         'attention_mask': attention_mask_batch,
#         'labels': labels_batch,
#         'alternative_labels': alternative_labels_batch
#     }
#
# def parse_string_to_data2(examples):
#     input_ids_batch = []
#     attention_mask_batch = []
#     labels_batch = []
#     alternative_labels_batch = []
#
#     for example in examples['text']:
#         # Удаляем теги <s> и </s>
#         example_string = example.strip('<s>').strip('</s>')
#         # Разделяем по тегу [INST] и [/INST]
#         input_part, outputs_part = example_string.split('[/INST]')
#         input_part = input_part.strip('[INST]').strip()
#         # Разделяем outputs по запятым
#         outputs = [output.strip() for output in outputs_part.split(',')]
#
#         # Токенизируем input и outputs
#         input_ids = input_part
#         labels = outputs[0]
#         if len(outputs) > 1:
#             alternative_labels = [tokenizer.encode(output, add_special_tokens=False) for output in outputs[1:]]
#         else:
#             alternative_labels = []
#
#         input_ids_batch.append(input_ids)
#         labels_batch.append(labels)
#         alternative_labels_batch.append(alternative_labels if alternative_labels else [[-100] * len(labels)])
#
#     max_len = max(max(len(ids) for ids in input_ids_batch),
#                   max(len(lbls) for lbls in labels_batch),
#                   max(max(len(alt) for alt in alts) for alts in alternative_labels_batch))
#
#     for idx in range(len(input_ids_batch)):
#         input_ids_batch[idx] += [tokenizer.pad_token_id] * (max_len - len(input_ids_batch[idx]))
#         attention_mask_batch[idx] += [0] * (max_len - len(attention_mask_batch[idx]))
#         labels_batch[idx] += [-100] * (max_len - len(labels_batch[idx]))
#         alternative_labels_padded = []
#         for alt in alternative_labels_batch[idx]:
#             alternative_labels_padded.append(alt + [-100] * (max_len - len(alt)))
#         alternative_labels_batch[idx] = alternative_labels_padded
#
#     return {
#         'input_ids': input_ids_batch,
#         'attention_mask': attention_mask_batch,
#         'labels': labels_batch,
#         'alternative_labels': alternative_labels_batch
#     }

# # Преобразование датасета
# parsed_data = dataset.map(parse_string_to_data, batched=True)

# DataCollator class
from typing import Any, Dict, List, Union
import torch
from transformers import DataCollatorForLanguageModeling


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm: bool = False):
        super().__init__(tokenizer, mlm=mlm)

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        print(features)
        texts = [feature['text'] for feature in features]
        alt_texts = []

        for i in range(2, 5):  # text2, text3, text4
            key = f'text{i}'
            alt_texts.append([feature.get(key, "") for feature in features])

        # Process texts and alt_texts
        all_texts = texts + [item for sublist in alt_texts for item in sublist if item]

        batch = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")

        return batch

# Пример использования
data_collator = CustomDataCollator(tokenizer=tokenizer)

# Настройки обучения
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Определение класса SFTTrainer с модифицированной передачей данных
class advanced_SFTTrainer(SFTTrainer):
    def __init__(self, model, args, data_collator=None, train_dataset=None, eval_dataset=None, tokenizer=None, model_init=None, label_smoother=None, **kwargs):
        super().__init__(
            model=model, args=args, data_collator=data_collator, train_dataset=train_dataset, eval_dataset=eval_dataset,
            tokenizer=tokenizer, model_init=model_init, **kwargs
        )
        self.label_smoother = label_smoother


    def compute_loss(self, model, inputs, return_outputs=False):
        print("COMPUTE LOSS ALTERNATIVE")
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        labels = inputs.get('labels')
        alternative_labels = inputs.get('alternative_labels')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits

        if self.label_smoother:
            main_loss = self.label_smoother(logits, labels)
        else:
            loss_fn = nn.CrossEntropyLoss()
            main_loss = loss_fn(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        if alternative_labels is not None and alternative_labels.nelement() != 0:
            alternative_losses = []
            for alt_labels in alternative_labels:
                if self.label_smoother:
                    alternative_loss = self.label_smoother(logits, alt_labels)
                else:
                    print("UNKNOWN!!")
                alternative_losses.append(alternative_loss)
            min_loss = torch.min(torch.stack([main_loss] + alternative_losses))
        else:
            min_loss = main_loss
        return (min_loss, outputs) if return_outputs else min_loss

# Создание и обучение модели
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

trainer = advanced_SFTTrainer(
    model=model, args=training_args, dataset_text_field="text", train_dataset=dataset, tokenizer=tokenizer, packing=True
)

#trainer = SFTTrainer(model=model, args=training_args, train_dataset=dataset, tokenizer=tokenizer, packing=True)

trainer.train()

