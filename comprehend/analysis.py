## Assumes steps 1 and 2 in the readme have been implemented
## Assumes the following files are in the ./experiments/results directory:
## 1. tok_wise_df.csv
## 2. tok_wise_tensors.pkl
## 3. prob_wise_rt.csv
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import numpy as np
import os
import argparse
from os import path
from tqdm import tqdm

import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle as pkl
from sklearn.metrics import cohen_kappa_score
from scipy import stats

from utils import plot_hist, plot_per_prob_hist, plot_zipf
from prompt import OpenAIInterface


# define the neural network architecture
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(input_size, output_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        return x
    

def train_model(model,
                loss_fn,
                optimizer,
                X, y, 
                num_epochs, 
                batch_size,
                device,
                verbose=False):

    # create the dataset and dataloader
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    model.zero_grad()

    # train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 5 == 0:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0

    return model

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN, (TP+TN)/(TP+FP+FN+TN)


def get_corrs(X, y, y_pred, top_k=None):
    # Poor design. Should ideally take X, y, top_k as inputs and return corrs
    if y_pred is not None:
        corr = np.corrcoef(y_pred, y)[0,1]
    else:
        corr = None

    xy = np.concatenate((X, np.expand_dims(y, axis=1)), axis=1)
    df = pd.DataFrame(xy)
    corrs = df.corr().to_numpy()
    # calculate the max correlation in the last column of corrs (the response)
    corr_raw = np.max(corrs[:-1,-1]).item()
    
    corrs_top_k_idxs = None
    if top_k is not None:
        assert top_k <= X.shape[1]
        # calculate the max correlation in the last column of corrs (the response)
        # but only for the top_k features
        corrs_top_k_idxs = np.argsort(corrs[:-1,-1])[-top_k:]
    
    return corr, corr_raw, corrs_top_k_idxs


def test_model(model, loss_fn, X_test, y_test, batch_size, verbose=False):
    # create the dataset and dataloader for testing
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # evaluate the trained model on the test set
    preds_all, labels_all, inputs_all = [], [], []

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs).squeeze()
            preds_all.append(outputs.detach())
            inputs_all.append(inputs.detach())
            labels_all.append(labels.detach())

    preds_all = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    inputs_all = torch.cat(inputs_all)
    
    diff_mse = loss_fn(preds_all, labels_all).item()
    corr_out, corr_input, _ = get_corrs(inputs_all, labels_all, preds_all)
    
    if verbose:
        print('Probability accuracy on test set: %.2f %%' % diff_mse)

    return diff_mse, corr_out, corr_input, preds_all


def predict_ridge(clf, X, y, top_k=None):
    if clf is not None:
        y_pred = clf.predict(X)
        diff_mse = mean_squared_error(y_pred, y)
    else:
        y_pred, diff_mse = None, None
    corr_out, corr_input, corrs_top_k_idxs = get_corrs(X, y, y_pred, top_k)
    return diff_mse, corr_out, corr_input, y_pred, corrs_top_k_idxs


def predict(tok_wise_df, 
            tok_wise_tensors, 
            prob_wise_rt, 
            train_set_frac=0.8,
            hidden_size=0, 
            output_size=1, 
            learning_rate=0.01, 
            num_epochs=50, 
            batch_size=32,
            out_dir='./',
            use_top_k_train_feats=False,
            top_k_train_feats=10,
            map_model='ridge',
            _alpha=10,
            seed=1024,):
    # Learn a model to predict the response for each token from the token-wise representation
    # and the token-wise log likelihood support
    
    # Column names in tok_wise_df:  'Token', 'LL Support', 'Response'
    
    # define device
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)

    identifier = map_model + "_" +\
                "alpha_"+str(_alpha) + "_" +\
                "topk_"+str(use_top_k_train_feats) + "_" +\
                str(hidden_size) + "_" +\
                str(output_size) + "_" +\
                str(learning_rate) + "_" +\
                str(num_epochs) + "_" + str(batch_size)

    # Remove entries in tok_wise_df['TokenID'] that are 0
    # Essentially, removing the first token (`def`) in each program
    tok_wise_df = tok_wise_df[tok_wise_df['TokenID'] != 0]

    frac = int(train_set_frac*tok_wise_df.shape[0])
    
    # Split data into train and test
    idx = np.random.permutation(tok_wise_df.shape[0])
    train_df = tok_wise_df.iloc[idx[:frac]]
    test_df = tok_wise_df.iloc[idx[frac:]]
    tok_wise_tensors = tok_wise_tensors.detach()
    train_tensors = tok_wise_tensors[idx[:frac]]
    test_tensors = tok_wise_tensors[idx[frac:]]

    # Convert responses to tensors
    y_train = torch.tensor(train_df['Response'].values, dtype=torch.float)
    y_test = torch.tensor(test_df['Response'].values, dtype=torch.float)

    loss_fn = nn.MSELoss()

    #X = torch.randn(1000, 10)
    #y = torch.randn(1000)
    #input_size = X.shape[1]
    #hidden_size = int(input_size/2)
    #model = train_model(X, y, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, device)
    
    cols_all = ['wordembed', 'LL'] # ['wordembed', 'LL'] # , ['wordembed', 'LL', 'LL_support', 'LL_support_diff']
    # best correl between wordembed's units and LL is 0.33. Range of correls between -0.31 to 0.33

    # get all combinations of cols_all
    cols_all_combos, all_preds = [], []

    for i in range(1, len(cols_all)+1):
        cols_all_combos.extend(list(itertools.combinations(cols_all, i)))

    for col in cols_all_combos:
        # get the columns to be used for training
        cols = list(col)
        
        # Make a string of the columns to be used for training
        _cols = ''
        for c in cols:
            _cols += c + '_'
        _cols = _cols[:-1]
        
        # if word_embed exists in cols, then add train_tensors to X_train
        if 'wordembed' in cols:
            cols.remove('wordembed')
            if len(cols) == 0:
                X_train = train_tensors
                X_test = test_tensors
            else:
                X_train = torch.cat((train_tensors, torch.tensor(train_df[cols].values, dtype=torch.float)), dim=1)
                X_test = torch.cat((test_tensors, torch.tensor(test_df[cols].values, dtype=torch.float)), dim=1)
        else:
            X_train = torch.tensor(train_df[cols].values, dtype=torch.float)
            X_test = torch.tensor(test_df[cols].values, dtype=torch.float)
            
        
        # Train the model on tok representations
        input_size = X_train.shape[1]
        hidden_size = int(input_size/2) + 1

        # create model and optimizer
        model = Net(input_size, hidden_size, output_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        if map_model == 'nn':
            model = train_model(model,
                    loss_fn,
                    optimizer,
                    X_train, y_train, 
                    num_epochs, batch_size, device)
            preds, corr, corr_raw, _ = test_model(model, loss_fn, X_test, y_test, batch_size)
            preds_train, corr_train, corr_raw_train, _ = test_model(model, loss_fn, X_train, y_train, batch_size)
        
        elif map_model == 'ridge':
            clf = Ridge(alpha=_alpha)
            X_train, X_test = X_train.detach().numpy(), X_test.detach().numpy()
            if use_top_k_train_feats and X_train.shape[1] > top_k_train_feats:
                preds_train, corr_train, corr_raw_train, _, top_k_feats = predict_ridge(None, X_train, y_train, top_k=top_k_train_feats)
                X_train = X_train[:, top_k_feats]
                X_test = X_test[:, top_k_feats]
            clf.fit(X_train, y_train.detach().numpy())
            preds, corr, corr_raw, _, _ = predict_ridge(clf, X_test, y_test.detach().numpy())
            preds_train, corr_train, corr_raw_train, _, _ = predict_ridge(clf, X_train, y_train.detach().numpy())
            
            # if y_test >= 0.75 then 1 else 0
            _y_train = torch.where(y_train >= 0.75, torch.tensor(1), torch.tensor(0))
            _y_test = torch.where(y_test >= 0.75, torch.tensor(1), torch.tensor(0))
            # Fit a binary logistic regression model
            clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
            clf.fit(X_train, _y_train.detach().numpy())
            _preds = clf.predict(X_test)
            TP, FP, TN, FN, acc = perf_measure(_y_test.detach().numpy(), _preds)


        all_preds.append((_cols, preds, corr, corr_raw, TP, FP, TN, FN, acc, preds_train, corr_train, corr_raw_train))
        print("{}:\n {}; {}".format(_cols, str(preds)[:5], str(corr)[:5]))
    
    all_preds = pd.DataFrame(all_preds, columns=['features', 'mse_test', 'corr_test', 'corr_in_test', 'TP', 'FP', 'TN', 'FN', 'acc', 'mse_train', 'corr_tr', 'corr_in_tr'])
    all_preds.to_csv(os.path.join(out_dir, 'predictions_{}.csv'.format(identifier)), index=False)


def get_per_problem_token_hist(df):
    # Group the responses by QuestionID, and select the top 3 Tokens for each QuestionID 
    # having the highest response
    # df1 = df.groupby('QuestionID').apply().reset_index(drop=True)
    
    # Find the number of unique tokens in the dataset that have a response greater than or equal to 0.5 for each QuestionID
    df2 = df.groupby('QuestionID').apply(lambda x: x.nlargest(5, 'Response')['Token'].nunique()).reset_index(name='num_tokens')

    # Find the maximum prediction corresponding to each QuestionID
    df3 = df.groupby('QuestionID').apply(lambda x: x['Response'].max()).reset_index(name='max_response')
    return df3


def prepare_pred_vector(all_preds, tok_df):
    ypred, y = [], []
    for qid, p in enumerate(all_preds):
        # Go through each token in tok_df for questionID = i
        # and create a new vector of predictions for each token
        # for each questionID. If the token is not present in the
        # prediction, then the prediction is 0
        preds_new = []
        toks_qid = tok_df[(tok_df['QuestionID'] == qid)]['Token']
        for t in toks_qid:
            # for each token tok in p, check if tok is in t.
            # if it is, preds_new.append(1), else preds_new.append(0)
            found = False
            for tok in p:
                if tok in t:
                    preds_new.append(1)
                    found = True
                    break
            if not found:
                preds_new.append(0)
            
        assert sum(preds_new) > 0
        assert len(preds_new) == len(toks_qid)
        ypred.extend(preds_new)
    # assign tok_df['Response'] to a list
    # if the value is greater than 0.8, then assign 1. Else, 0
    y = [1 if r >= 0.75 else 0 for r in tok_df['Response']]
    return ypred, y


def predict_gpt4_prompts(tok_df, 
                         prompts, 
                         key_path, 
                         few_shot=5, 
                         out_dir='output', 
                         identifier='gpt4', 
                         all_prompts_path='./data/all_prompts_responses.csv'):    
    head_prompt = prompts[0]
    # remove the head prompt from the prompts list
    prompts = prompts[1:]
    all_preds = []

    if few_shot == -1:
        # if all_prompts exists, load from file.
        # else, create all_prompts and save to file
        if os.path.exists(all_prompts_path):
            _all_preds = pd.read_csv(all_prompts_path, keep_default_na=False).values.tolist()
            all_preds = []
            for p in _all_preds:
                temp = []
                for t in p:
                    if t == '':
                        continue
                    else:
                        temp.append(t)
                all_preds.append(temp)
        else:
            # for each program in prompts, prepare a prompt text comprising all other programs except the current program
            # call the openai interface to get the predictions for the current program with the prompt text
            # store the predictions in a dataframe
            personal_api_key = open(key_path, 'r').read().strip()
            openai_interface = OpenAIInterface(api_key=personal_api_key)
            # openai_interface.predict_text('I am a dog. What are you?', mode='chat-gpt4')
            for i in range(len(prompts)):
                prompt = head_prompt
                for j in range(len(prompts)):
                    if i != j:
                        prompt += prompts[j]+"\n"
                prompt_parts = prompts[i].split("@@@")
                curr_prompt, solution_text, curr_solution = prompt_parts[0], prompt_parts[1], prompt_parts[2]
                prompt += (curr_prompt + "@@@" + solution_text)
                # print('prompt: ', prompt)
                # print('program: ', prompts[i])
                preds = openai_interface.predict_text(prompt, mode='chat-gpt4')
                each_preds = preds.split('\n')
                toks = []
                for e in each_preds:
                    toks.extend(e.split(' '))
                all_preds.append(toks)
            pd.DataFrame(all_preds).to_csv(all_prompts_path, index=False)

    y_pred, y = prepare_pred_vector(all_preds, tok_df)
    
    df = pd.DataFrame([y_pred, y]).T
    df.columns = ['pred_{}'.format(identifier), 'pred_human']
    df.to_csv(os.path.join(out_dir, 'predictions_{}.csv'.format(identifier)), index=False)

    diff_mse = mean_squared_error(y_pred, y)
    corr = np.corrcoef(y_pred, y)[0,1]
    TP, FP, TN, FN, acc = perf_measure(y, y_pred)
    # save [diff_mse, corr, TP, FP, TN, FN, acc] as columns of a dataframe in a csv
    df = pd.DataFrame([diff_mse, corr, TP, FP, TN, FN, acc]).T
    df.columns=['diff_mse', 'corr', 'TP', 'FP', 'TN', 'FN', 'acc']
    df.to_csv(os.path.join(out_dir, 'prediction_stats_{}.csv'.format(identifier)), index=False)
    return all_preds, diff_mse, corr, TP, FP, TN, FN, acc


def get_inter_rater_stats(rater_tok_wise_df, out_dir):
    # for each unique questionID, select all columns with the name Response_*
    # compute the cohen_kappa_score for each pair of raters
    
    # get all unique questionIDs
    qids = rater_tok_wise_df['QuestionID'].unique()
    
    # get all columns with the name Response_*
    cols = [c for c in rater_tok_wise_df.columns if 'Response_' in c]
    
    # Get the mean and std number of toks per QuestionID
    lens = []
    for qid in qids:
        lens.append(len(rater_tok_wise_df[(rater_tok_wise_df['QuestionID'] == qid)]))
    m, s = np.mean(lens), np.std(lens)

    # for each questionID, compute the cohen_kappa_score for each pair of raters
    # and store the scores in a list
    scores_kappa, scores_correl = [], []
    for qid in qids:
        temp1, temp2 = [], []
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                rater1 = rater_tok_wise_df[(rater_tok_wise_df['QuestionID'] == qid)][cols[i]].values.tolist()
                rater2 = rater_tok_wise_df[(rater_tok_wise_df['QuestionID'] == qid)][cols[j]].values.tolist()
                temp1.append(cohen_kappa_score(rater1, rater2))
                temp2.append(stats.pearsonr(rater1, rater2).statistic)

        scores_kappa.append(temp1)
        scores_correl.append(temp2)
    
    # for each questionID, compute the correl of each rater with the mean of all other raters
    # the other raters should not include the current rater
    # store the scores in a list
    scores_correl_normalized = []
    for qid in qids:
        temp2 = []
        for i in range(len(cols)):
            rater1 = rater_tok_wise_df[(rater_tok_wise_df['QuestionID'] == qid)][cols[i]].values.tolist()
            other_raters = [rater_tok_wise_df[(rater_tok_wise_df['QuestionID'] == qid)][c].values.tolist() for c in cols if c != cols[i]]
            mean_other_raters = np.mean(np.array(other_raters), axis=0)
            temp2.append(stats.pearsonr(rater1, mean_other_raters).statistic)
        scores_correl_normalized.append(temp2)
    
    
    scores_correl = np.array(scores_correl)
    all_norm_scores = np.array(scores_correl_normalized)
    
    # drop any row containing nan
    scores_correl = scores_correl[~np.isnan(scores_correl).any(axis=1)]
    all_norm_scores = all_norm_scores[~np.isnan(all_norm_scores).any(axis=1)]

    # compute the mean
    mean_scores_correl = np.mean(scores_correl)
    mean_scores_correl_norm = np.mean(all_norm_scores)

    #################################################
    ## across prob. correl
    scores_correl_across_probs = []
    _temp = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            rater1 = rater_tok_wise_df[cols[i]].values.tolist()
            rater2 = rater_tok_wise_df[cols[j]].values.tolist()
            _temp.append(stats.pearsonr(rater1, rater2).statistic)        
        scores_correl_across_probs.append(_temp)
    
    scores_correl_normalized_across_probs = []
    for i in range(len(cols)):
        rater1 = rater_tok_wise_df[cols[i]].values.tolist()
        other_raters = [rater_tok_wise_df[c].values.tolist() for c in cols if c != cols[i]]
        mean_other_raters = np.mean(np.array(other_raters), axis=0)
        scores_correl_normalized_across_probs.append(stats.pearsonr(rater1, mean_other_raters).statistic)
    mean_scores_correl_normalized_across_probs = np.mean(scores_correl_normalized_across_probs)

    return mean_scores_correl_norm, mean_scores_correl_normalized_across_probs

def get_prob_wise_correls(tok_wise_df, tok_reprs, y=None):
    qids = tok_wise_df['QuestionID'].unique()
    ll_corrs, emb_corrs = [], []
    for qid in qids:
        # Get the corresponding row numbers where QuestionId is qid
        row_nums = tok_wise_df[(tok_wise_df['QuestionID'] == qid)].index.tolist()
        
        # Select Response from tok_wise_df where QuestionID is qid
        if y is not None:
            response_i = np.array(y[row_nums])
        else:
            response_i = np.array(tok_wise_df[(tok_wise_df['QuestionID'] == qid)]['Response'].values.tolist())

        ll_i = np.array(tok_wise_df[(tok_wise_df['QuestionID'] == qid)]['LL'].values.tolist())
        ll_i = np.expand_dims(ll_i, axis=1)
        
        # Calculate the correlation between response_i and LL
        _, c1, _ = get_corrs(ll_i, response_i, None, None)
        _, c2, _ = get_corrs(tok_reprs[row_nums,:], response_i, None, 1)

        ll_corrs.append(c1)
        emb_corrs.append(c2)
    
    r1, r2 = sum(ll_corrs)/len(ll_corrs), sum(emb_corrs)/len(emb_corrs)

    # Select Response from tok_wise_df where QuestionID is qid
    if y is not None:
        response_i_across_probs = np.array(y)
    else:
        response_i_across_probs = np.array(tok_wise_df['Response'].values.tolist())

    ll_i_across_probs = np.array(tok_wise_df['LL'].values.tolist())
    ll_i_across_probs = np.expand_dims(ll_i_across_probs, axis=1)
    
    # Calculate the correlation between response_i and LL
    _, c1, _ = get_corrs(ll_i_across_probs, response_i_across_probs, None, None)
    _, c2, _ = get_corrs(tok_reprs, response_i_across_probs, None, 1)


    return r1, r2, c1, c2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path', action='store', default='./experiments/results/', help='Out dir')
    parser.add_argument('--get_hists', default=1, type=int, help='get histograms of token representations')
    parser.add_argument('--get_preds', default=1, type=int, help='get predictions from token representations')

    opt = parser.parse_args()
    out_dir = opt.out_path
    get_hists = bool(opt.get_hists)
    get_preds = bool(opt.get_preds)

    # Load data data
    gpt4_id = 'gpt4'
    rater_tok_wise_df = pd.read_csv('./experiments/results/rater_tok_wise_df.csv')
    tok_wise_df = pd.read_csv('./experiments/results/tok_wise_df.csv')
    tok_wise_tensors = torch.load('./experiments/results/tok_wise_tensors.pt')
    prob_wise_rt = pd.read_csv('./experiments/results/prob_wise_rt.csv')
    prompts = open('./data/prompts.txt').read().strip().split('$$$')
    key_path = './data/openai_api_key_sam.key'
    all_prompts = './data/all_prompts_responses.csv'
    # if gpt4 predictions exist, read csv
    if os.path.exists('./experiments/results/predictions_gpt4.csv'):
        gpt4_preds = pd.read_csv('./experiments/results/predictions_gpt4.csv')

    if get_hists:
        r11, r12, c11, c12 = get_prob_wise_correls(tok_wise_df, tok_wise_tensors, gpt4_preds['pred_{}'.format(gpt4_id)].values)
        # r21, r22, c21, c22 = get_prob_wise_correls(tok_wise_df, tok_wise_tensors, gpt4_preds['pred_human'].values)
        r31, r32, c31, c32 = get_prob_wise_correls(tok_wise_df, tok_wise_tensors)
        # get_prob_wise_correls(tok_wise_df, tok_wise_tensors)
        c1, c2 = get_inter_rater_stats(rater_tok_wise_df, out_dir=out_dir)

        plot_per_prob_hist(tok_wise_df, out_dir=out_dir)
        hists = get_per_problem_token_hist(tok_wise_df)
        plot_hist(hists, out_dir=out_dir)
        plot_zipf()
        
        
        
    if get_preds:
        predict_gpt4_prompts(tok_wise_df, prompts, key_path, few_shot=-1, out_dir=out_dir, identifier=gpt4_id, all_prompts_path=all_prompts)
        predict(tok_wise_df, tok_wise_tensors, prob_wise_rt, out_dir=out_dir)
