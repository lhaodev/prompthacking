import re
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch

tokenizer_roberta = RobertaTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate"
)
model_roberta = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate"
)
model_roberta.eval()


def is_hateful(text, threshold=0.5):
    inputs = tokenizer_roberta(text, return_tensors="pt")
    outputs = model_roberta(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probabilities[0, 1].item() > threshold


def prompt_injection_detected(text):
    # Add any specific keywords, phrases, or patterns that might indicate prompt injection
    prompt_injection_patterns = [r"ignore", r"pwned"]

    for pattern in prompt_injection_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def prompt_leaking_detected(text):
    # Add any specific keywords, phrases, or patterns that might indicate prompt leaking
    prompt_leak_patterns = [
        r"prompt",
        r"user_input",
        r"instructions",
        r"instruction",
        r"assistant",
        r"user",
        r"output",
    ]

    for pattern in prompt_leak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


sentence = "i think writing is good for your soul"

if not prompt_injection_detected(sentence):
    if not is_hateful(sentence):
        tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
        model_gpt2 = GPT2LMHeadModel.from_pretrained(
            "gpt2", pad_token_id=tokenizer_gpt2.eos_token_id
        )
        input_ids = tokenizer_gpt2.encode(sentence, return_tensors="pt")
        output = model_gpt2.generate(
            input_ids,
            max_length=1000,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
        sentence = tokenizer_gpt2.decode(output[0], skip_special_tokens=True)
        if not prompt_leaking_detected(sentence):
            print(sentence)
        else:
            print(
                "Prompt leaking detected in the generated text. Generating new output..."
            )
    else:
        print("The input sentence is hateful. Please provide a different input.")

else:
    print(
        "The input sentence may contains prompt injection. Please provide a different input."
    )
