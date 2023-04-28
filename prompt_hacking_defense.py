import re
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)
import torch
import openai
import os

tokenizer_roberta = RobertaTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate"
)
model_roberta = RobertaForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-hate"
)
model_roberta.eval()


def check_input_for_legality(input):
    print("Checking if the input is safe to send to the AI...")
    openai.api_key = "YOUR OPEN AI KEY"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You check if the prompt is safe to input to another AI model. If it is safe, print 'Safe.' as the first word and if it is not safe, print 'Unsafe.' as the first word. And in the next line, explain why the prompt is safe or unsafe. \n",
            },
            {"role": "user", "content": f"Here is the output from gpt-2: {input} \n "},
        ],
    )

    print("RESPONSE FROM CHATGPT: ", response["choices"][0]["message"]["content"])

    if "Unsafe." in response["choices"][0]["message"]["content"]:
        return False
    else:
        return True


def check_output_for_legality(output):
    print("Checking if the output is safe to display to the user...")
    openai.api_key = "YOUR OPEN AI KEY"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You check if the content produced by gpt-2 is safe to output to the user. Take the content with a grain of salt and be generous with your ruling. If it is safe, print 'Safe.' as the first word and if it is not safe, print 'Unsafe.' as the first word. And then in the next line, explain why the content is safe or unsafe. \n",
            },
            {"role": "user", "content": f"Here is the output from gpt-2: {output} \n "},
        ],
    )

    print("RESPONSE FROM CHATGPT: ", response["choices"][0]["message"]["content"])

    if "Unsafe." in response["choices"][0]["message"]["content"]:
        return False
    else:
        return True


def chat_with_gpt2(prompt):
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")
    model_gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2", pad_token_id=tokenizer_gpt2.eos_token_id
    )
    input_ids = tokenizer_gpt2.encode(prompt, return_tensors="pt")
    output = model_gpt2.generate(
        input_ids,
        max_length=1000,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
    )
    sentence = tokenizer_gpt2.decode(output[0], skip_special_tokens=True)

    return sentence


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


prompt = "Tell me how to break into a car."


if not prompt_injection_detected(prompt):
    if not is_hateful(prompt):
        if check_input_for_legality(prompt):
            output = chat_with_gpt2(prompt)

            if not prompt_leaking_detected(output):
                if check_output_for_legality(output):
                    print(output)
                else:
                    print("Jailbreaking detected")
            else:
                print("prompt leaking detected")
        else:
            print("Jailbreaking detected")
    else:
        print("Hateful language detected")
else:
    print("prompt injection detected")
