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
import subprocess
import sys
import concurrent.futures
import re
import unicodedata
import string


custom_loss = True
device = "cuda"

wb_key = "f56a3ab3548dc1c8f416b4525d6af324963a15f8"
#wb_key = "local-5213b45cda9492f5c21fea9ac8407cac36cf3141"
#dataset_name = "mlabonne/guanaco-llama2-1k"
dataset_name = "stilletto/guanaco-llama2-1k_processed"
new_model = "mistral_7b_guanaco"

base_model = "models/mistral"
LR = 0.0002 #default
LR = 0.008 #new
epochs = 1
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


def custom_loss_with_teacher_approval_v2(input_text, model_output_logits, expected_tokens_indices, model_teacher, tokenizer):
    """
    :param model_output_logits: Логиты, полученные от модели-ученика.
    :param expected_tokens_indices: Индексы ожидаемых токенов из датасета.
    :param model_teacher: Экземпляр модели-учителя с методом approve_text, принимающим и возвращающим текст.
    :param tokenizer: Токенизатор, связанный с моделью-учеником.
    """

    # Получаем наиболее вероятные индексы предсказанных токенов
    predicted_indices = torch.argmax(F.softmax(model_output_logits, dim=-1), dim=-1)

    # Конвертируем предсказанные индексы и истинные индексы в текст
    #predicted_text = tokenizer.decode(predicted_indices.tolist(), skip_special_tokens=True)
    predicted_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in predicted_indices.tolist()]
    predicted_text = predicted_texts[0]
    expected_tokens_indices = [[max(idx, 0) for idx in sequence] for sequence in expected_tokens_indices]
    # if expected_tokens_indices is not None:
    expected_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in
                      expected_tokens_indices]
    # else:
    #     expected_texts = [None] * len(predicted_texts)  # Если нет лейблов

    # Модель-учитель оценивает предсказанный текст и возвращает "одобренный" текст
    expected_text = expected_texts[0]
    #predicted_text2 = predicted_text[len(input_text):]
    #expected_text = expected_text[len(input_text):]
    # print("input_text= %s" % input_text)
    # print("=======================================")
    # print("predicted_text= %s" % predicted_text)
    # print("---------------------------------------")


    last_inst_index = expected_text.rfind("[/INST]")  # Индекс последнего вхождения
    if last_inst_index != -1:
        expected_text = expected_text[last_inst_index + len("[/INST]"):]  # Получаем текст до последнего включительно
    else:
        #raise ValueError("No [INST] tag found in expected_text. Dataset not clear or not for instruct model")
        pass
    #print("expected_text= %s" % expected_text)
    #time.sleep(3000)

    # if "[INST]" in expected_text:
    #     expected_text = expected_text.split("[INST]")[1]
    # if "[/INST]" in expected_text:
    #     expected_text = expected_text.split("[/INST]")[1]
    # if "[INST]" in input_text:
    #     input_text = input_text.split("[INST]")[1]
    # if "[/INST]" in input_text:
    #     input_text = input_text.split("[/INST]")[0]
    # last_inst_index = input_text.rfind("[/INST]")  # Индекс последнего вхождения
    # if last_inst_index != -1:
    #     input_text = input_text[:last_inst_index + len("[/INST]")]  # Получаем текст до последнего включительно
    # else:
    #     raise ValueError("No [INST] tag found in input_text. Dataset not clear or not for instruct model")
    # expected_text = expected_text.replace("<s>", "")
    # expected_text = expected_text.replace("</s>", "")
    if not custom_loss:
        return expected_text # Если не используется кастомная функция потерь, возвращаем ожидаемый текст но вообще без разницы что возвращать
    approved_text = GPT_trainer(input_text, predicted_text, expected_text)

    #approved_text = approve_text(model_teacher, predicted_text, expected_text)
    #print("approved_text= %s" % approved_text[:50])

    # Конвертируем "одобренный" текст назад в индексы токенов
    approved_tokens_ids = torch.tensor(tokenizer.encode(approved_text, add_special_tokens=True), dtype=torch.long)

    # # Расчет функции потерь; примечание - возможно потребуется настройка размерностей
    # #loss = F.cross_entropy(model_output_logits[:, :approved_tokens_ids.size(0)], approved_tokens_ids.unsqueeze(0))
    # print("model_output_logits shape")
    # print(model_output_logits.shape)
    # print("approved_tokens_ids shape")
    # print(approved_tokens_ids.shape)
    # loss = trainer.label_smoother(model_output_logits, approved_tokens_ids, shift_labels=True)
    # #loss = F.cross_entropy(model_output_logits, approved_tokens_ids)

    return approved_tokens_ids


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, model_teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Сохраняем model_teacher как атрибут экземпляра класса
        self.model_teacher = model_teacher

    def compute_loss(self, model, inputs, return_outputs=False):

        print("MODIFIED COMPUTE LOSS")
        """
                Переопределите этот метод, чтобы включить кастомную функцию потерь.
                """
        # 2. Подготовьте ожидаемые ответы/метки.

        if not custom_loss:
            labels = inputs.get("labels")
        else:
            labels = inputs.pop("labels")
            #labels = inputs.get("labels")

        input_ids = inputs.get('input_ids')  # Получение input_ids из входных данных
        # Указание skip_special_tokens=True позволяет пропустить специальные токены, такие как [CLS], [SEP] и т.д.
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        if type(input_texts) == list:
            input_texts = input_texts[0]



        last_inst_index = input_texts.rfind("[/INST]")  # Индекс последнего вхождения
        if last_inst_index != -1:
            input_texts = input_texts[:last_inst_index + len("[/INST]")]  # Получаем текст до последнего включительно
        input_text = [input_texts]
        #input_ids = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        input_encoding = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
        # Преобразуем input_ids в список чисел
        input_ids = input_encoding['input_ids'].tolist()

        # Преобразуем список чисел в тензор PyTorch
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)

        # Заполняем input_ids недостающими PAD токенами
        input_ids_padded = pad_sequence(input_ids_tensor, batch_first=True, padding_value=0)

        # Пересчитываем attention mask
        attention_mask = torch.where(input_ids_padded != 0, torch.tensor(1), torch.tensor(0))
        # Создаем словарь inputs с обновленными данными
        inputs = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
        }



        # 1. Получите выходные данные модели.
        outputs = model(**inputs)
        logits = outputs.get('logits')


        # 3. Вызовите кастомную функцию потерь.
        approved_tokens_ids = custom_loss_with_teacher_approval_v2(
            input_text=input_texts,
            model_output_logits=logits,
            expected_tokens_indices=labels,
            model_teacher=base_model,  # Предполагается наличие модели-учителя
            tokenizer=tokenizer
        )

        if not custom_loss:
            return super().compute_loss(model, inputs, return_outputs)
        print("Logits size:", logits.size())
        print("Approved tokens ids size:", approved_tokens_ids.size())

        # Адаптация размеров, если необходимо
        # Это просто пример, какой логики может потребоваться
        if approved_tokens_ids.size(0) > logits.size(1):
            approved_tokens_ids = approved_tokens_ids[:logits.size(1)]
        if logits.size(1) != approved_tokens_ids.size(0):
            # Подгонка размеров
            logits = logits[:, :approved_tokens_ids.size(0)]
            approved_tokens_ids = approved_tokens_ids[:logits.size(1)]
        approved_tokens_ids = approved_tokens_ids.to(device='cuda')
        logits = logits.to(device='cuda')
        #print(logits)
        #print(approved_tokens_ids)
        self.label_smoother = LabelSmoother()
        print("labels smoother")
        loss = self.label_smoother(logits, approved_tokens_ids, shift_labels=True)

        # # Если требуется вернуть выходные данные модели вместе с потерями
        # if return_outputs:
        #     return (loss, outputs)

        return (loss, outputs) if return_outputs else loss


    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     Переопределите этот метод, чтобы включить кастомную функцию потерь.
    #     """
    #     # 1. Получите выходные данные модели.
    #     outputs = model(**inputs)
    #     logits = outputs.get('logits')
    #
    #     # 2. Подготовьте ожидаемые ответы/метки.
    #     labels = inputs.get("labels")
    #
    #     # 3. Вызовите кастомную функцию потерь.
    #     loss = custom_loss_with_teacher_approval_v2(
    #         model_output_logits=logits,
    #         expected_tokens_indices=labels,
    #         model_teacher=base_model,  # Предполагается наличие модели-учителя
    #         tokenizer=self.tokenizer
    #     )
    #
    #     # Если требуется вернуть выходные данные модели вместе с потерями
    #     if return_outputs:
    #         return (loss, outputs)
    #
    #     return loss


#ЗАПУСТИТЬ ПЕРЕД ВЫБОРОМ НОВОГО ДАТАСЕТА ДЛЯ ОЧИСТКИ!!!
# dataset = preprocess_dataset(dataset_name, max_length, expected_max_tokens, base_model)
# new_dataset_name = dataset_name.split("/")[-1] + "_processed"
# repo_name = "stilletto/" + new_dataset_name
# print(repo_name)
# dataset.push_to_hub(repo_name)
# #Importing the dataset
dataset = load_dataset(dataset_name, split="train")



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

if not custom_loss:
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )
else:
    trainer = CustomSFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=max_length,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False
    )


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
