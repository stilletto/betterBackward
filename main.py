import time
import os
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
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

#import wandb
from datasets import load_dataset
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

def GPT_trainer(input_text,predicted_text,ground_truth_text):
    print("input_text= %s"%input_text[:20])
    print("predicted_text= %s"%predicted_text[:20])
    print("ground_truth_text= %s"%ground_truth_text[:20])
    client = OpenAI(api_key="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")

    question = str({"input_text": input_text,
                    "predicted_text": predicted_text,
                    "ground_truth_text": ground_truth_text})

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "Your task is to evaluate the text according to the following rules. You will have the ground_truth_text and the predicted_text. If the predicted_text is an adequate part of the ground_truth_text, simply repeat it. If the last word or part of a word of the predicted_text is incorrect, then change the text so that it is as similar as possible in the arrangement of words and their choice to the predicted_text but in meaning is similar to the ground_truth_text. That is, correct it, trying to minimally change the predicted_text. But if the predicted_text is correct, including the correct part, do not change it, do not answer anything unnecessary, just return it as is. Examples:\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Москв\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is correct predicted part of text. Because it is part of right answer \"Москва\" . You must to return \"Москв\"\n\nIncorrect:\n {\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Москву\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is incorrect predicted part of text. Because last latter is wrong. You must to return \"Москва\" because it is most close right answer\n\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Насколько я знаю Москва\",\n\"ground_truth_text\": \"Самый большой город России - Москва\"\n}\nIt is correct predicted text or part of text. Because it is content right. The sentence is constructed differently, but the meaning is the same and it is correct . You must to return \"Насколько я знаю Москва\"\n\nCorrect:\n{\"input_text\": \"Какой самый большой город России?\",\n\"predicted_text\": \"Мо\",\n\"ground_truth_text\": \"Самый большой город России - Москва\",\n}\nIt is correct predicted part of text. Because it is part of right answer \"Москва\". You must to return \"Мо\"\n\nCorrect:\n{\"input_text\": \"Какой самый большой город Юпитера?\",\n\"predicted_text\": \"Не\",\n\"ground_truth_text\": \"На Юпитере нет городов.\"\n}\nIt is correct predicted part of text. Because further the phrase can be predicted as, for example, “I don’t know” or “There are no cities on Jupiter” and these will be the correct answers, so we consider this the correct beginning of the phrase.  You must to return \"Не\"\n\n\nIncorrect:\n{\"input_text\": \"Какой самый большой город Юпитера?\",\n\"predicted_text\": \"..i\",\n\"ground_truth_text\": \"На Юпитере нет городов.\"\n}\nIt is incorrect predicted part of text.  This is the wrong beginning of a phrase, no matter what the next phrase is, the correct answer can hardly begin like that. Therefore, return the same token in the account, which is the last in this case. Since the correct first one would be \"На\". You must to return \"На\"\n"
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
    print(response)
    return str(response)

# model = AutoModelForCausalLM.from_pretrained(
#     "mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True
# )
wb_key = "f56a3ab3548dc1c8f416b4525d6af324963a15f8"

dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "mistral_7b_guanaco"

LR = 0.0002
epochs = 10
#wandb.login(key=wb_key)
# print("wandb itinializing")
# wandb.init(
#     # set the wandb project where this run will be logged
#     project="bbw",
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": LR,
#     "architecture": "CNN",
#     "dataset": dataset_name,
#     "epochs": epochs,
#     }
# )
#
# print("wandb initialized")
# time.sleep(100)

#Importing the dataset
#base_model = "C:\\LocalRepo\\betterBackward\\models\\Mistral-7B-v0.1"
base_model = "mistralai/Mistral-7B-v0.1"
dataset = load_dataset(dataset_name, split="train")
dataset["text"][100]
#
def approve_text(model_teacher, predicted_text, expected_text):
    """
    :param model_teacher: Экземпляр модели-учителя с методом approve_text, принимающим и возвращающим текст.
    :param predicted_text: Предсказанный текст.
    :param expected_text: Ожидаемый текст.
    :return: Одобренный текст.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_teacher, device_map="auto", load_in_4bit=True
    )
    model_path = "models/Mistral-7B-v0.1"
    model_path = "C:\\LocalRepo\\betterBackward\\models\\Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    ask_to_approve = "Do you approve of this text? (yes/no): "
    model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("cuda")


    generated_ids = model.generate(**model_inputs)
    tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
    model_inputs = tokenizer(
        ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
    ).to("cuda")
    generated_ids = model.generate(**model_inputs)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(output)
    print("done----------")



    expected_text = predicted_text
    # Предположим, что модель-учитель просто возвращает ожидаемый текст
    return expected_text

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
    print("predicted_texts= %s"%predicted_texts)
    predicted_text = predicted_texts[0]
    print("predicted_text= %s"%predicted_text)
    print("expected_tokens_indices= %s"%expected_tokens_indices)
    print("expected_tokens_indices.tolist()= %s"%expected_tokens_indices.tolist())
    expected_tokens_indices = [[max(idx, 0) for idx in sequence] for sequence in expected_tokens_indices]
    if expected_tokens_indices is not None:
        expected_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in
                          expected_tokens_indices]
    else:
        expected_texts = [None] * len(predicted_texts)  # Если нет лейблов

    # Модель-учитель оценивает предсказанный текст и возвращает "одобренный" текст
    expected_text = expected_texts[0]
    print("expected_text= %s"%expected_text)
    approved_text = GPT_trainer(input_text, predicted_text, expected_text)
    #approved_text = approve_text(model_teacher, predicted_text, expected_text)

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
        # 1. Получите выходные данные модели.
        outputs = model(**inputs)
        logits = outputs.get('logits')
        print("logits shape")
        print(logits.shape)

        # 2. Подготовьте ожидаемые ответы/метки.
        #labels = inputs.get("labels")
        labels = inputs.pop("labels")
        print("labels shape")
        print(labels.shape)


        input_ids = inputs.pop('input_ids')  # Получение input_ids из входных данных
        # Указание skip_special_tokens=True позволяет пропустить специальные токены, такие как [CLS], [SEP] и т.д.
        input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # 3. Вызовите кастомную функцию потерь.
        approved_tokens_ids = custom_loss_with_teacher_approval_v2(
            input_text=input_texts,
            model_output_logits=logits,
            expected_tokens_indices=labels,
            model_teacher=base_model,  # Предполагается наличие модели-учителя
            tokenizer=self.tokenizer
        )
        print("Logits size:", logits.size())
        print("Approved tokens ids size:", approved_tokens_ids.size())

        # Адаптация размеров, если необходимо
        # Это просто пример, какой логики может потребоваться
        if logits.size(1) != approved_tokens_ids.size(0):
            # Подгонка размеров
            logits = logits[:, :approved_tokens_ids.size(0)]
        approved_tokens_ids = approved_tokens_ids.to(device='cuda')
        print(logits)
        print(approved_tokens_ids)
        self.label_smoother = LabelSmoother()
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

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
print("1111111111111111")
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
print("2222222222222222")
model = get_peft_model(model, peft_config)
print("3333333333333333")
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
    report_to=[]
    #report_to="wandb"
)
print("4444444444444444")

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     max_seq_length= None,
#     dataset_text_field="text",
#     tokenizer=tokenizer,
#     args=training_arguments,
#     packing= False,
# )

trainer = CustomSFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)



print("5555555555555555")
trainer.train()
print("6666666666666666")
trainer.model.save_pretrained(new_model)
print("7777777777777777")
# wandb.finish()
model.config.use_cache = False


prompt = "How do I find true love?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])



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
