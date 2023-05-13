import pickle as pkl
import argparse
import os
import pandas as pd
import torch
from models import get_models
from tqdm.contrib import tzip
from os import path
from dataset import get_dataset_fn


def get_model_ll_support(text, 
                model, 
                tokenizer, 
                config, 
                infer_interval,
                top_p, 
                top_k, 
                max_len, 
                num_return_seqs, 
                sample_info, 
                output_dir
            ):

    txt, result = [], []
    for cnt, t in enumerate(text.split(' ')):
        txt.append(t)
        if cnt%infer_interval == 0:
            curr = ' '.join(txt)
            inputs = tokenizer.encode(curr, return_tensors="pt")
            '''
            outputs = model(inputs, output_hidden_states=True)
            '''
            outputs = model.generate(inputs=inputs, 
                                    num_return_sequences=num_return_seqs,
                                    # max_length=max_len,
                                    top_p=top_p,
                                    top_k=top_k,
                                    do_sample=False,
                                    return_dict_in_generate=True,
                                    output_scores=True
                                    )
            
            _outputs = []
            _outputs.append(curr)
            for o in outputs.sequences:
                decoded = tokenizer.decode(o, skip_special_tokens=True)
                # print(decoded)
                _outputs.append(decoded)
            per_token_support = torch.column_stack(outputs.scores[1])
            
            # Normalize by the number of tokens in the vocab
            per_token_support = per_token_support / config.vocab_size

            per_sent_support = torch.sum(per_token_support, dim=1).tolist()
            _outputs.extend(per_sent_support)
            result.append(_outputs)
    pd.DataFrame(result).to_csv(path.join(output_dir, '{}.csv'.format(sample_info)), index=False)
    return result


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


def align_qualtrics_tokenizer_model_tokenizer(text, first_token, input_ids, tokenizer):
    '''
    The purpose of the function is to tokenize the text input according to Qualtric's tokenization,
    and map the resulting tokens to their corresponding indices in the 
    input_ids using the provided tokenizer. 
    Specifically, it tokenizes the text input into a list of individual words (splitting on spaces), 
    and then compares each word with the tokens in input_ids to find the indices where each word
    starts and ends.

    The function then returns two outputs: map_space_tok_to_tokenizer_tok and li_space_tok_to_tokenizer_tok. 
    The map_space_tok_to_tokenizer_tok is a dictionary that maps each space-separated 
    word in text to a list of corresponding indices in input_ids. 
    The li_space_tok_to_tokenizer_tok is a list of lists, where each sublist contains the indices
    in input_ids that correspond to each space-separated word in text.
    '''
    _text_space = text.split(' ')
    qualtrics_toks = []
    for t in _text_space:
        if t != '':
            qualtrics_toks.append(t)
  
    model_toks = []
    for i in input_ids:
        model_toks.append((tokenizer.decode(i), i))
    
    assert first_token.strip(" ") in qualtrics_toks[0]
    _qualtrics_toks = qualtrics_toks[1:]

    current_tok, cnt, tok_idxs, tok_pos = '', 0, [], []
    tot_toks = 0
    list_space_tok_to_tokenizer_tok = []
    
    for c, (tok, tok_id) in enumerate(model_toks):
        if tok == '':
            continue
        current_tok += tok.strip(" ")
        tot_toks += 1
        tok_idxs.append(tok_id)
        tok_pos.append(c)
        if cnt < len(_qualtrics_toks) and current_tok == _qualtrics_toks[cnt]:
            # map_space_tok_to_tokenizer_tok[cnt] = tok_idxs
            list_space_tok_to_tokenizer_tok.append((_qualtrics_toks[cnt], tok_idxs, tok_pos))
            current_tok = ''
            cnt += 1
            tok_idxs = []
            tok_pos = []
    
    # The total tokens should be equal to the number of tokens in the input_ids
    assert tot_toks == len(model_toks)
    return list_space_tok_to_tokenizer_tok


def get_model_representation_of_input(text, 
                model, 
                tokenizer, 
                config,
                sample_info, 
                output_dir):
    
    '''
    text = """def sum_list(li):
        sum = 0
        for i in li:
            sum += i
        return sum
    """
    '''
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(inputs['input_ids'], output_hidden_states=True)
    hs = outputs.hidden_states[-1].squeeze()
    
    # Remove outputs corresponding to special tokens
    li = inputs['input_ids'].squeeze().tolist()
    toks_to_retain = []
    for c, input_id in enumerate(li):
        if input_id not in tokenizer.all_special_ids:
            toks_to_retain.append(c)
    
    input_ids = inputs['input_ids'][:, toks_to_retain]
    hs = hs[toks_to_retain, :]
    outputs_logits = outputs.logits.squeeze()[toks_to_retain, :]

    probs = torch.log_softmax(outputs_logits, dim=1).detach()

    assert input_ids.shape[1] == hs.shape[0] == outputs_logits.shape[0]

    # Align input_ids with probs.
    # probs_i has the likelihood of the tokens occurring at (i+1)
    # See https://discuss.huggingface.co/t/announcement-generation-get-probabilities-for-generated-output/30075/17
    input_ids_sample = input_ids[:, 1:]
    hs_sample = hs[1:, :]
    probs_sample = probs[:-1, :]
    
    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    gen_probs = torch.gather(probs_sample, 1, input_ids_sample.squeeze()[:, None]).squeeze(-1)

    first_token = tokenizer.decode(input_ids[:,0])
    list_space_sep_idxs = align_qualtrics_tokenizer_model_tokenizer(text, 
                        first_token,
                        input_ids_sample.squeeze(), 
                        tokenizer
                        )
    
    tok_to_rep, tok_to_ll = [], []
    for (tok, tok_idxs, tok_pos) in list_space_sep_idxs:
        # Find the average embedding of all tokens corresponding to a word
        tok_to_rep.append((tok, tok_idxs,hs_sample[tok_pos].mean(dim=0).detach()))
        tok_to_ll.append((tok, tok_idxs, gen_probs[tok_pos].min(dim=0).values.detach()))

    # Append to the front of tok_to_rep as many zero tensors as skipped tokens
    tok_to_rep.insert(0, (first_token, [], torch.zeros(config.hidden_size)))
    tok_to_ll.insert(0, (first_token, [], torch.tensor(0.)))

    torch.save([tok_to_rep, tok_to_ll, list_space_sep_idxs, text], open(path.join(output_dir, '{}.pkl'.format(sample_info)), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', action='store', default='refactory', help='Dataset name')
    parser.add_argument('--dataset_path', action='store', default='../refactory/', help='Path to train data')
    parser.add_argument('--number_of_records', action='store', 
                        dest='number_of_records', default=1, type=int, help='Number of records to analyze from the dataset')
    parser.add_argument('--expt_dir', action='store', default='./experiments', help='Path to directory where experiment results are saved')
    parser.add_argument('--batch_size', action='store', default=8, type=int)
    parser.add_argument('--iter', default=5, type=int, help='Number of optimizer iterations')
    parser.add_argument('--max_len', default=128, type=int, help='Max input length')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--multinomial_samples', default=1, type=int, help='Number of random multinomial samples to draw')
    parser.add_argument('--model_names', action='store', nargs='+', default=['santa-coder'])
    parser.add_argument('--top_p', default=0.9, type=float, help='top p probability threshold')
    parser.add_argument('--top_k', default=0, type=int, help='top K')
    parser.add_argument('--infer_interval', default=15, type=int, help='Interval for inference')
    parser.add_argument('--max_len_output', default=128, type=int, help='max length of model output')
    parser.add_argument('--num_return_seqs', default=1, type=int, help='num of return sequences')
    parser.add_argument('--mode', default=1, type=int, help='mode of operations')
    # parser.add_argument('--mask_idxs', action='store', dest='mask_idxs', nargs='+', default=[2], type=int)

    opt = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mode =  opt.mode # 1: get model representations; 2: get model LL support sizes
    model_names = opt.model_names # ['codeberta-small', 'codeberta-base-mlm', 'plbart']
    # mask_idxs   = opt.mask_idxs # 1: random masks; 2: masks only on those lines with diffs

    codes, sample_info = get_dataset_fn(opt.dataset_name)(pth=opt.dataset_path, max_len=opt.max_len, number_of_files=int(opt.number_of_records))
    models = get_models(model_names)
    
    os.makedirs(os.path.join(opt.expt_dir, opt.dataset_name), exist_ok=True)
    identifier = '{}_{}_{}_{}'.format(opt.infer_interval, opt.top_p, opt.max_len_output, opt.dataset_name)

    for (model, tokenizer, config), model_name in zip(models, model_names):
        cnt = 0
        for c, s in tzip(codes, sample_info):
            sample_identifier = '{}_{}_{}'.format(model_name, s, identifier)
            if mode == 1:
                get_model_representation_of_input(c, 
                                         model, 
                                         tokenizer, 
                                         config, 
                                         sample_identifier, 
                                         os.path.join(opt.expt_dir, opt.dataset_name)
                                        )
            elif mode == 2:
                get_model_ll_support(c, 
                            model, 
                            tokenizer, 
                            config, 
                            opt.infer_interval,
                            opt.top_p, 
                            opt.top_k, 
                            opt.max_len_output, 
                            opt.num_return_seqs,
                            sample_identifier, 
                            os.path.join(opt.expt_dir, opt.dataset_name)
                            )
