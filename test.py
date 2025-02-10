#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import tensorflow as tf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TFAutoModelForCausalLM,
)

# تعریف یک کلاس Dataset سفارشی برای آماده‌سازی داده‌های متنی
class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, block_size=64):
        self.examples = []
        for text in texts:
            # توکنایز کردن متن
            tokenized = tokenizer(text, truncation=True, max_length=block_size, return_tensors="pt")
            # حذف بعد اضافی (batch dimension)
            self.examples.append({k: v.squeeze(0) for k, v in tokenized.items()})
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def main():
    # 1. بارگذاری توکنایزر و مدل GPT-2 (نسخه PyTorch)
    model_name = "gpt2"  # مدل پایه
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # اگر توکن pad تنظیم نشده باشد، آن را به عنوان توکن پایان جمله (eos) در نظر می‌گیریم
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 2. آماده‌سازی دیتاست کوچک (برای مثال‌های آموزشی)
    texts = [
        "سلام، حال شما چطور است؟",
        "امروز هوا زیباست و پرندگان آواز می‌خوانند.",
        "من عاشق یادگیری درباره هوش مصنوعی و یادگیری ماشین هستم.",
        "مدل‌های Transformer ابزارهای قدرتمندی برای پردازش زبان طبیعی هستند.",
        "یادگیری عمیق بسیاری از زمینه‌های فناوری را متحول کرده است.",
    ]
    
    dataset = CustomTextDataset(texts, tokenizer, block_size=64)
    
    # 3. آماده‌سازی Data Collator برای مدل‌سازی زبانی (بدون استفاده از masked language modeling)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 4. تنظیمات آموزش برای Trainer (PyTorch)
    training_args = TrainingArguments(
        output_dir="./gpt2-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=1,                # برای مثال تنها 1 epoch
        per_device_train_batch_size=2,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5,
    )
    
    # 5. ایجاد شیء Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 6. آموزش مدل
    print("آموزش مدل با استفاده از PyTorch در حال شروع است...")
    trainer.train()
    
    # 7. ذخیره مدل و توکنایزر
    model_save_path = "./gpt2-finetuned"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # 8. تولید متن با استفاده از مدل PyTorch
    prompt_text = "معنای زندگی چیست"
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
    output_ids = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("\nمتنی که توسط مدل PyTorch تولید شده:")
    print(generated_text)
    
    # 9. بارگذاری نسخه TensorFlow مدل (از همان دایرکتوری ذخیره شده)
    print("\nدر حال بارگذاری نسخه TensorFlow مدل...")
    tf_model = TFAutoModelForCausalLM.from_pretrained(model_save_path)
    
    # 10. تولید متن با استفاده از مدل TensorFlow
    input_ids_tf = tokenizer.encode(prompt_text, return_tensors="tf")
    tf_output_ids = tf_model.generate(
        input_ids_tf,
        max_length=50,
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )
    tf_generated_text = tokenizer.decode(tf_output_ids[0], skip_special_tokens=True)
    print("\nمتنی که توسط مدل TensorFlow تولید شده:")
    print(tf_generated_text)

if __name__ == "__main__":
    main()
