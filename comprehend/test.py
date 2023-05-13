import argparse
from pprint import pprint
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
# from model_outputs import get_word_id_tok_idx


def get_word_id_tok_idx(encoded):
    desired_output = []
    for word_id in encoded.word_ids():
        if word_id is not None:
            start, end = encoded.word_to_tokens(word_id)
            if start == end - 1:
                tokens = [start]
            else:
                tokens = [start, end-1]
            if len(desired_output) == 0 or desired_output[-1] != tokens:
                desired_output.append(tokens)
    return desired_output


def to_tokens_and_logprobs(model, tokenizer, input_texts):
    inputs = tokenizer(input_texts, return_tensors="pt")
    input_ids = inputs.input_ids
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    wlist1 = get_word_id_tok_idx(inputs)
    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            # if token not in tokenizer.all_special_ids:
            text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    return batch


def test_llm():
    model_name = "huggingface/CodeBERTa-small-v1"
    model_name = "gpt2"
    model_name = "bigcode/santacoder"
    #model_name = "Salesforce/codet5-base"

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        
    
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    input_texts1 = ["One plus one is two"]
    input_texts2 = ["Good morning"]
    input_texts3 = ["Hello, how are you?"]
    input_texts4 = ["Banana is a fruit"]
    input_texts5 = ["""def sum_list(li):
        sum = 0
        for i in li:
            sum += i
        return sum
    """]

    batch = to_tokens_and_logprobs(model, tokenizer, input_texts5)
    pprint(batch)


if __name__ == '__main__':
    choice = 1
    if choice == 1:
        test_llm()