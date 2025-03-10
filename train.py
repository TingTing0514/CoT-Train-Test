import wandb
import torch
from trl import SFTTrainer
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

device = torch.device("npu")


model_name = "DeepSeek-R1-Distill-Qwen-7B"
max_seq_length = 2048
dtype = None
load_in_4bit = True
output_dir = f"/root/CoT-Train-Test/outputData/{model_name}-v1/outputs"
run = wandb.init(
    project='Fine-tune-DeepSeek-R1-Distill-Qwen-7B on law-reasoning-SFT',
    job_type="training"
)

model, tokenizer = AutoModelForCausalLM.from_pretrained(
    model_name = "/root/CoT-Train-Test/DeepSeek-R1-Distill-Qwen-7B",
    max_length=max_seq_length,
    load_in_4bit=load_in_4bit
).to(device), AutoTokenizer.from_pretrained(model_name)


train_prompt_style = """以下是描述一项任务的说明，以及提供进一步背景信息的输入内容。
请给出恰当的回答以完成请求。
在回答之前，请仔细思考问题，并构建一个逐步的思维链条，以确保回答合乎逻辑且准确无误
### 您是一位合同法专家。
请回答以下法律问题。
### Question:
{}

### Response:
<think>
{{
}}
</think>
{{
}}
"""

question = "我和朋友合伙开了一家咖啡店，我们签了一份合作协议，但现在他突然说要退出，不想继续合作了。我该怎么办？"


EOS_TOKEN = tokenizer.eos_token

def format_prompts_func(examples):
    inputs = examples["input"]
    cots = examples["reasoning"]
    outputs = examples["output"]
    texts = []
    for input_question, cot, output in zip(inputs, cots, outputs):
        text = train_prompt_style.format(input_question, cot, output) + EOS_TOKEN
        texts.append(text)
    return {
        "text": texts
    }

train_dataset = load_dataset(
    path="json",
    data_files="/root/CoT-Train-Test/CoT-Train-Test/law_CoT.json",
)
train_dataset = train_dataset.map(format_prompts_func, batched=True)

# If using LoRA, you would manually configure it without `unsloth` here
from peft import get_peft_model, LoRAConfig

peft_config = LoRAConfig(
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,
    random_state=3407
)

model = get_peft_model(model, peft_config)

train_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    num_train_epochs=1,
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

output_model_dir = "/root/CoT-Train-Test/DeepSeek-R1-Distill-Qwen-7B-law-CoT-v1"
trainer.save_model(output_model_dir)
# 也保存 tokenizer
tokenizer.save_pretrained(output_model_dir)
