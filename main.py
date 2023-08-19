import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    line(
        "line-corporation/japanese-large-lm-3.6b",
        "ゆゆ式で一番かわいいのは"
    )


def line(llm, prompt):
    # トークナイザーとモデルの準備
    tokenizer = AutoTokenizer.from_pretrained(
        llm,
        use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_safetensors=False
    )

    # 推論の実行
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    tokens = model.generate(
        input_ids.to(device=model.device),
        min_length=50,
        max_length=300,
        temperature=1.0,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    output = tokenizer.decode(tokens[0])
    print(output)


if __name__ == '__main__':
    main()
