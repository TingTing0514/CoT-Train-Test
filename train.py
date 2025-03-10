import wandb
from unsloth import FastLanguageModel,is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset


model_name = "DeepSeek-R1-Distill-Qwen-1.5B"
max_seq_length = 2048
dtype = None
load_in_4bit = True
output_dir = f"/root/data/outputData/{model_name}-v2/outputs"
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Qwen-1.5B on medical_o1-reasoning-SFT',
    job_type="training"
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/root/data/DeepSeek-R1-Distill-Qwen-1.5B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit
)




train_prompt_style = """以下是描述一项任务的说明，以及提供进一步背景信息的输入内容。
请给出恰当的回答以完成请求。
在回答之前，请仔细思考问题，并构建一个逐步的思维链条，以确保回答合乎逻辑且准确无误
### 您是一位在临床推理、诊断和治疗计划方面拥有高级知识的医学专家。
请回答以下医学问题。
### Question:
{}

### Response:
<think>
{}
</think>
{}
"""

question = "一位 61 岁女性，长期在咳嗽或打喷嚏等活动时出现尿失禁，但夜间无尿失禁现象。该患者接受了妇科检查和棉签试验。根据这些检查结果，膀胱测压最有可能显示其残余尿量和逼尿肌收缩情况如何？"


# FastLanguageModel.for_inference(model) 
# inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

# outputs = model.generate(
#     input_ids=inputs.input_ids,
#     attention_mask=inputs.attention_mask,
#     max_new_tokens=1200,
#     use_cache=True,
# )
# response = tokenizer.batch_decode(outputs)
# print(response[0].split("### Response:")[1])

EOS_TOKEN = tokenizer.eos_token

def format_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    texts = []
    for input_question,cot,output in zip(inputs,cots,outputs):
        text = train_prompt_style.format(input_question,cot,output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts
    }

train_dataset = load_dataset(
    path="json",
    data_files="/root/data/dataset/medical-o1-reasoning-SFT/medical_o1_sft_Chinese.json",
    split="train[0:20000]",  

    )
train_dataset = train_dataset.map(format_prompts_func, batched = True)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, #低秩矩阵的维度
    target_modules=[
        "q_proj",  #查询投影矩阵
        "k_proj",  #键投影矩阵 
        "v_proj",  #值投影矩阵
        "o_proj",  #输出投影矩阵
        "gate_proj", #MLP（多层感知机）层的投影矩阵
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16, #缩放因子 默认值
    lora_dropout=0, #不应用 dropout，即所有神经元都参与训练
    bias="none", #LoRA 适配器的偏置（bias）设置 
    use_gradient_checkpointing="unsloth", #梯度检查点（Gradient Checkpointing）的设置
    random_state=3407, #随机种子
    use_rslora=False, # RSLoRA（Randomized Sparse LoRA） 的设置
    loftq_config=None #LoFTQ 是一种结合量化和低秩分解的技术，用于进一步压缩模型
)

train_args= TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        # Use num_train_epochs = 1, warmup_ratio for full training runs!
        num_train_epochs=1,
        # warmup_steps=5,
        # max_steps=60,
        warmup_ratio=0.1,
        save_steps=1000,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
    )


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=train_args
)
trainer.train()

output_model_dir = "/root/data/DeepSeek-R1-1.5B-Medical-COT-v2"
trainer.save_model(output_model_dir)
# 也保存 tokenizer
tokenizer.save_pretrained(output_model_dir)