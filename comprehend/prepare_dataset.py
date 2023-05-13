import argparse
import pandas as pd
import os
import torch


def filter_cols(cols, columns):
    cols = [c for c in cols if c in columns]
    return cols


def count_entries(participant_responses):
    responses, responses_count = {}, {}
    participant_responses = participant_responses[participant_responses.notnull()]
    for r in participant_responses:
        r1 = r.replace(',,', '$,') # $ does not come up in Python-permissible alphabet set
        key_vals = r1.split(',')
        for kv1 in key_vals:
            kv = kv1.replace('$', ',')
            keys, vals = kv.split(':')[0], ':'.join(kv.split(':')[1:])
            vals = vals.strip(' ')
            
            # If keys are not ints, then a comma has been parsed incorrectly
            try:
                keys = int(keys)
            except:
                continue

            if keys in responses:
                if responses[keys] != vals:
                    assert abs(len(responses[keys]) - len(vals)) == 1
                    # assert that the difference in length is due to a comma
                    if len(responses[keys]) > len(vals):
                        assert vals == responses[keys].replace(',', '')
                        new_val = responses[keys]
                    else:
                        assert responses[keys] == vals.replace(',', '')
                        new_val = vals
                    responses[keys] = new_val
                    # print('Assert:{}:{}'.format(responses[keys], vals))
            else:
                responses[keys] = vals
            
            if keys in responses_count:
                responses_count[keys] += 1
            else:
                responses_count[keys] = 1
    
    # Normalize
    for k in responses_count.keys():
        responses_count[k] = responses_count[k] / len(participant_responses)

    responses_tok_count = {}
    for k, v in responses.items():
        assert responses_count[k] >= 0 and responses_count[k] <= 1
        responses_tok_count[v] = responses_count[k]

    return responses_tok_count


def consolidate_responses(df):
    '''
        Consolidate responses from multiple columns into a single column
    '''
    _df = df.apply(count_entries, axis=1)
    return _df


def consolidate_rt(df):
    '''
        Consolidate RTs from multiple columns into a single column
    '''
    _df = df.apply(lambda x: x.median(skipna=True))
    return _df


def load_participant_responses(path, max_responses=10, max_participants=10):
    df = pd.read_excel(path)
    
    # Filter out rows for which "Finished" is "True"
    df = df[df['Finished'] == 'True']
    
    # Prepare list of column names to select
    start_from = 4
    col_responses, col_descrip, col_rt = [], [], []

    for _ in range(max_responses):
        responses = 'Q{}_1'.format(start_from)
        rt = 'Q{}_Page Submit'.format(start_from + 1)
        descrip = 'Q{}'.format(start_from + 2)
        start_from += 3
        col_responses.extend([responses])
        col_descrip.extend([descrip])
        col_rt.extend([rt])
    
    # Remove column names from list that do not exist in the dataframe
    col_responses = filter_cols(col_responses, df.columns)
    col_descrip = filter_cols(col_descrip, df.columns)
    col_rt = filter_cols(col_rt, df.columns)

    # Select columns
    # Rows: questions; Columns: participants
    df_descrip = df[col_descrip].T
    _df_rt = df[col_rt].T
    _df_responses = df[col_responses].T

    # Consolidate responses
    df_responses = consolidate_responses(_df_responses)

    # Consolidate RTs
    df_rt = consolidate_rt(_df_rt)

    per_rater_response, per_rater_rt = [], []
    for i in range(len(_df_responses.columns)):
        # Consolidate responses
        t1 = consolidate_responses(_df_responses.iloc[:, i].to_frame())
        per_rater_response.append(t1)

        # Consolidate RTs
        t2 = consolidate_rt(_df_rt.iloc[:, i].to_frame())
        per_rater_rt.append(t2)

    return df_responses, df_rt, df_descrip, per_rater_response, per_rater_rt


def align_participant_responses_token_wise_data(participant_responses=None, 
                                                tok_ll_support_sz=None, 
                                                id_reps=None,
                                                id_ll=None):
    df_responses, prob_wise_rt = participant_responses[0], participant_responses[1]

    # Check number of rows in df_responses is the same as the len of tok_ll_support_sz, tok_reps
    assert len(df_responses) == len(tok_ll_support_sz) == len(id_reps)

    tok_wise_df, tok_wise_tensors = None, []

    # Iterate over each problem
    for i in range(len(id_reps)):
        reps_i = id_reps[i]
        ll_i = id_ll[i]
        toks_i = [r[0] for r in reps_i]
        tok_ll_support_sz_i = tok_ll_support_sz[i]
        response_i = df_responses[i]
        
        # Check if all tokens in tok_reps_i are the same as that in tok_ll_support_sz_i
        assert set(toks_i) == set(tok_ll_support_sz_i.keys())

        entries, prev = [], None
        for cnt, k in enumerate(toks_i):
            k_strip = k.strip()
            if k_strip not in response_i: # The keys in response_i are stripped by Qualtrics
                response = 0
            else:
                response = response_i[k_strip]

            if prev is not None:
                diff = tok_ll_support_sz_i[k] - prev
            else:
                diff = 0
            
            entries.append([i, cnt, k, ll_i[cnt][2].data.tolist(), tok_ll_support_sz_i[k], diff, response])
            tok_wise_tensors.append(reps_i[cnt][2].data)
            prev = tok_ll_support_sz_i[k]
        
        # The number of non-zero responses should be the same as the len of response_i
        # assert len([e[-1] for e in entries if e[-1] != 0]) == len(response_i)
        if set([e[2].strip() for e in entries if e[-1] != 0]) != set(response_i.keys()):
            print("assert:\n{}\n".format(set(response_i.keys()).difference(set([e[2].strip() for e in entries if e[-1] != 0]))))

        # Convert entries into a df
        df_responses_i = pd.DataFrame(entries, columns=['QuestionID', 'TokenID', 'Token', 'LL', 'LL_support', 'LL_support_diff', 'Response'])

        # Append df_responses_i to df_responses_all
        if tok_wise_df is None:
            tok_wise_df = df_responses_i
        else:
            tok_wise_df = pd.concat([tok_wise_df, df_responses_i], ignore_index=True)
    
    # Convert entries_tensors_all into a tensor
    tok_wise_tensors = torch.stack(tok_wise_tensors, dim=0)

    assert tok_wise_tensors.shape[0] == tok_wise_df.shape[0]

    return tok_wise_df, tok_wise_tensors, prob_wise_rt


def get_token_wise_ll_support_data(path, outpath):
    '''
        Get token wise log likelihood support data
    '''
    problemid_idx_tok_ll_map = []
    for file in sorted(os.listdir(path), reverse=False):
        # print(file)
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            '''
            Select the first column and store it in a list
            '''
            cumulative_toks = df[df.columns[0]].tolist()
            ll_support = df[df.columns[2]].tolist()
            idx_tok_ll_map, cnt = {}, 1
            prev = ''
            for i, c in enumerate(cumulative_toks):
                trimmed = c.strip(' ')
                if trimmed == prev:
                    continue
                tok = trimmed.split(' ')[-1]
                # tok_llsupport_map[tok] = ll_support
                idx_tok_ll_map[tok] = ll_support[i]
                prev = trimmed
                cnt += 1
            problemid_idx_tok_ll_map.append(idx_tok_ll_map)

    return problemid_idx_tok_ll_map #, tok_llsupport_map


def get_token_wise_representations(path, outpath):
    problemid_reps, problemid_ll, problem_texts = [], [], []
    for file in sorted(os.listdir(path), reverse=False):
        # print(file)
        if file.endswith(".pkl"):
            rep = torch.load(os.path.join(path, file))
            problemid_reps.append(rep[0])
            problemid_ll.append(rep[1])
            problem_texts.append(rep[3])

    return problemid_reps, problemid_ll, problem_texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses_path', action='store', help='File path to participant responses')
    parser.add_argument('--token_wise_ll_support_path', action='store', help='Path to train data')
    parser.add_argument('--token_wise_representations_path', action='store', help='Path to representations data')
    parser.add_argument('--out_path', action='store', help='Path to store results')
    
    opt = parser.parse_args()

    df_responses, df_rt, df_descrip, per_rater_response, per_rater_rt = load_participant_responses(opt.responses_path)
    idx_tok_ll_map = get_token_wise_ll_support_data(opt.token_wise_ll_support_path, opt.out_path)
    idx_rep, idx_ll, idx_texts = get_token_wise_representations(opt.token_wise_ll_support_path, opt.out_path)
    
    # Get rater-wise, token-wise data
    rater_tok_wise_df = pd.DataFrame()
    for r in range(len(per_rater_response)):
        _tok_wise_df, _tok_wise_tensors, _prob_wise_rt = align_participant_responses_token_wise_data(participant_responses=[per_rater_response[r], per_rater_response[r]], 
                                                    tok_ll_support_sz=idx_tok_ll_map, 
                                                    id_reps=idx_rep,
                                                    id_ll=idx_ll)
        # Concatenate the column Response from the dataframes _tok_wise_df in rater_tok_wise_df
        if r == 0:
            rater_tok_wise_df = _tok_wise_df
            # Change column name to Response_r
            rater_tok_wise_df = rater_tok_wise_df.rename(columns={'Response': 'Response_{}'.format(r)})
        else:
            resp = _tok_wise_df['Response'].to_frame()
            # Change column name to Response_r
            resp = resp.rename(columns={'Response': 'Response_{}'.format(r)})
            rater_tok_wise_df = pd.concat([rater_tok_wise_df, resp], axis=1)
    rater_tok_wise_df.to_csv(os.path.join(opt.out_path, 'rater_tok_wise_df.csv'), index=False)

    # Get token-wise data with consolidated rater responses
    tok_wise_df, tok_wise_tensors, prob_wise_rt = align_participant_responses_token_wise_data(participant_responses=[df_responses, df_rt], 
                                                tok_ll_support_sz=idx_tok_ll_map, 
                                                id_reps=idx_rep,
                                                id_ll=idx_ll)
    tok_wise_df.to_csv(os.path.join(opt.out_path, 'tok_wise_df.csv'), index=False)
    torch.save(tok_wise_tensors, os.path.join(opt.out_path, 'tok_wise_tensors.pt'))
    # Add idx_texts as a column to the dataframe prob_wise_rt
    prob_wise_rt.to_csv(os.path.join(opt.out_path, 'prob_wise_rt.csv'), index=False)