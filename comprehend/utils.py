import pandas as pd
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
import copy
import numpy as np
import copy


def plot_zipf():
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5.5})
    plt.style.use('seaborn-deep')
    plt.rcParams['figure.figsize'] = [12.0, 8.0]
    plt.rcParams['figure.dpi'] = 500

    fntsz = 24
    lblsz = 28
    titlesz = 30
    lblpd = -20
    tksz = 24
    lgndsz = 20

    # tokens contains two indices: 0 and 1.
    # tokens[0] contains the list of first 20 words appearing in the English dictionary
    # tokens[1] contains the list of first 20 tokens appearing in Python code
    tokens = [['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',    # English
                'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'],
                ['if', 'for', 'while', 'def', 'return', 'in', 'range', 'print', 'len', 'import',    # Python
                'True', 'False', 'None', 'and', 'or', 'not', 'elif', 'else', 'break', 'continue']]
    # responses contains two indices: 0 and 1.
    # responses[0] contains the list of first 20 responses for the tokens in tokens[0]
    # responses[1] contains the list of first 20 responses for the tokens in tokens[1]
    zipf = [0.9, 0.8, 0.7, 0.4, 0.35, 0.3,     # English
                0.25, 0.2, 0.1, 0.09, 0.08, 0.07,
                0.06, 0.05, 0.04, 0.03, 0.02, 0.01,
                0.009, 0.008]
    # gaussian contains a list of 20 random gaussian samples sampled from mean 0.5 and std 0.1
    gaussian = [0.5 + np.random.normal(0, 0.1) for i in range(20)]
    gaussian = sorted(gaussian, reverse=True)
    
    # responses has two copies of zipf
    responses = [[copy.deepcopy(zipf), copy.deepcopy(zipf)], 
                 [copy.deepcopy(gaussian), copy.deepcopy(gaussian)]
                ]

    # Add small random noise to responses
    for i in range(len(responses)):
        for j in range(len(responses[i])):
            responses[i][j] = [r + np.random.normal(0, 0.01) for r in responses[i][j]]
    
    descriptions = ['English', 'Python']
    
    # plot the bar chart
    for cnt, resp in enumerate(responses):
        if cnt == 0:
            id = 'Zipf'
        else:
            id = 'Gaussian'
        for c, (t, r, d) in enumerate(zip(tokens, resp, descriptions)):
            if c == 0:
                color = '#009FBD'
                hatch = 'o'
            else:
                color = '#FFA500'
                hatch = '//'
            plt.bar(t, r, 
                        color=color,
                        width=0.6, 
                        hatch=hatch)
            # Add a smooth line connecting the centers of each bar
            plt.plot([i for i in range(len(t))], savgol_filter(r, 5, 3), color='black', linewidth=3)
            
            plt.title("Token distribution in {}".format(d), fontsize=titlesz)
            plt.xlabel('\nUnique tokens\n'.format(len(responses)), fontsize=lblsz, labelpad=lblpd)
            plt.ylabel('\nFrequency of occurrence', fontsize=lblsz)
            plt.ylim(0.0, 1.0)
            plt.xticks(rotation=40, fontsize=tksz)
            plt.yticks(fontsize=tksz)
            # Remove top and right borders
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.savefig("zipf_{}_{}.png".format(d, id), bbox_inches='tight')
            plt.close()


def plot_per_prob_hist(hists, out_dir):
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5.5})
    plt.style.use('seaborn-deep')
    plt.rcParams['figure.figsize'] = [12.0, 8.0]
    plt.rcParams['figure.dpi'] = 500

    fntsz = 24
    lblsz = 28
    titlesz = 30
    lblpd = -20
    tksz = 18
    lgndsz = 20

    # get max QuestionID from hists. 
    # hists is a dataframe with columns: QuestionID, Response, Token
    max_QuestionID = max(hists['QuestionID'])
    len_50, len_50_norm = [], []
    # for each problem in df, plot the histogram of tokens whose Response > 0
    for i in range(max_QuestionID+1):
        # get the Response and Tokens corresponding to QuestionID i
        df_i = hists[hists['QuestionID'] == i]
        tot_toks = len(df_i)

        # select those Tokens whose Response > 0
        df_i = df_i[df_i['Response'] > 0]
        
        # clip the Tokens length to 10 chars
        df_i['Token'] = df_i['Token'].apply(lambda x: x[:10])

        # sort the df in descending order of Response
        df_i = df_i.sort_values(by=['Response'], ascending=False)
        
        # get the Response and Token columns
        responses, tokens = df_i[['Response']].iloc[:,0].tolist(), df_i[['Token']].iloc[:,0].tolist()

        plt.bar(tokens, responses, 
                 color='#009FBD',
                 width=0.6, 
                 hatch='//')
        plt.title("TRR distribution in problem {}. (Total toks={})".format(i, tot_toks), fontsize=titlesz)
        plt.xlabel('\nTokens with TRR>0 (N={})\n'.format(len(responses)), fontsize=lblsz, labelpad=lblpd)
        plt.ylabel('\nToken response rate (TRR)', fontsize=lblsz)
        plt.yticks(np.arange(0, 1.1, 0.1), fontsize=tksz)
        plt.xticks(rotation=40)
        plt.savefig(out_dir + "/problem_{}.png".format(i), bbox_inches='tight')
        plt.close()

        # select those tokens whose response > 0.5
        df_i = df_i[df_i['Response'] > 0.5]
        len_50.append(len(df_i)/tot_toks)
        len_50_norm.append(len(df_i)/len(responses))
    
    avg_len_50 = sum(len_50)/len(len_50)
    avg_len_50_norm = sum(len_50_norm)/len(len_50_norm)
    qq

def plot_hist(hists, out_dir):
    # From the dataframe hists, plot the histograms of the column max_response
    # and save the plot in the directory out_dir
    # Ensure the plot is aesthetically pleasing, and preferably uses seaborn
    # for plotting
    # sns.set_context("poster")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5.5})
    plt.style.use('seaborn-deep')
    plt.rcParams['figure.figsize'] = [12.0, 8.0]
    plt.rcParams['figure.dpi'] = 500

    fntsz = 24
    lblsz = 28
    titlesz = 30
    lblpd = -20
    tksz = 18
    lgndsz = 20
    
    print(hists['max_response'])

    plt.hist(hists['max_response'], 
             bins=np.arange(0.1, 1.2, 0.1)-0.05, 
             color='#009FBD', 
             edgecolor='black', 
             rwidth=0.6, 
             hatch='//')
    #plt.bar(range(0, 0.1, 1), counts, width=0.8, align='center')
    plt.title("Distribution of max TRR per program", fontsize=titlesz)
    plt.xlabel('\nMax token response rate (TRR)\n', fontsize=lblsz, labelpad=lblpd)
    plt.ylabel('\n# of programs (N={})\n'.format(len(hists)), fontsize=lblsz, labelpad=lblpd)
    plt.xticks(np.arange(0.1, 1.1, 0.1), fontsize=tksz)
    plt.yticks(np.arange(6), fontsize=tksz)
    plt.savefig(out_dir+'hist.png', bbox_inches='tight')
    plt.clf()


def plot(hists, out_dir):
    # sns.set_context("poster")
    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 5.5})
    plt.style.use('seaborn-deep')
    plt.rcParams['figure.figsize'] = [12.0, 8.0]
    plt.rcParams['figure.dpi'] = 500

    fntsz = 24
    lblsz = 28
    titlesz = 30
    lblpd = -20
    tksz = 18
    lgndsz = 20

    pth = sys.argv[1] # shash/data/results/'

    ### fMRI task
    pth2 = './results/final_data/consolidated_for_plots_fmri_senti_expt.xlsx'
    df = pd.read_excel(pth2, sheet_name="combined")
    # df = pd.concat([df['pred'], df['base_filtered_pred'], df['drive_filtered_pred']], axis=1)
    # df = df.dropna()
    # df1 = pd.concat([df['pred'], df['base_filtered_pred']], axis=1)
    # df1 = df1.dropna()
    # s1 = df1['pred']
    # s2 = df1['base_filtered_pred'].dropna()
    s1 = df['pred']
    s2 = df['base_filtered_pred'].dropna()
    s3 = df['drive_filtered_pred'].dropna()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)

    plt.hist([s1, s2, s3], 
                bins=140, 
                range=(-0.8, 1.2), 
                histtype='step', 
                color=['#377eb8', '#f781bf', '#ff7f00'],
                linewidth=2.5
            )

    ax = plt.gca()
    patches = []
    for pp in ax.patches:
        patches.append(copy.copy(pp.get_xy()))
    plt.clf()
    for pp in patches:
        yhat = savgol_filter(pp[:,1], 51, 3)
        plt.plot(pp[:,0], yhat)

    plt.title("fMRI task", fontsize=titlesz)
    plt.xlabel('\n$y^\mathrm{pred}$\n', fontsize=lblsz, labelpad=lblpd)
    plt.ylabel('\n# of sentences\n', fontsize=lblsz, labelpad=lblpd)
    plt.xticks(fontsize=tksz)
    plt.yticks(fontsize=tksz)
    plt.legend(['Our: max $y^\mathrm{desired}$', 'Our: min $y^\mathrm{desired}$', 'SBM'], 
                loc="upper left", 
                fontsize=lgndsz, 
                frameon=False
                )

    # handles, labels = plt.gca().get_legend_handles_labels()
    # print(labels)
    # order = [2,1,0]
    # plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

    plt.savefig(pth+'hist_fmri.png', bbox_inches='tight')
    plt.clf() 

    ### Senti task
    pth1 = './results/final_data/senti_synth_results_consolidated.csv'
    df = pd.read_csv(pth1)
    df_preds = pd.concat([
    df[(df['targeted index']==1) & (df['status']=='orig')]['prediction0'].reset_index(drop=True),
    df[(df['targeted index']==1) & (df['status']=='orig')]['prediction1'].reset_index(drop=True),
    df[(df['targeted index']==0) & (df['status']=='synth')]['prediction0'].reset_index(drop=True),
    df[(df['targeted index']==1) & (df['status']=='synth')]['prediction1'].reset_index(drop=True),
    ], axis=1)
    df_preds.columns = ['orig_pred0', 'orig_pred1', 'synth_pred0', 'synth_pred1']

    # ax = df_preds.plot.hist(bins=100, alpha=0.5)
    # fig = ax.get_figure()
    plt.hist([df_preds['orig_pred0'], df_preds['orig_pred1'], df_preds['synth_pred0'], df_preds['synth_pred1']],
                bins=140,
                histtype='step', 
                color=['#377eb8', '#984ea3', '#f781bf', '#ff7f00'],
                linewidth=2
            )

    ax = plt.gca()
    patches = []
    for pp in ax.patches:
        patches.append(copy.copy(pp.get_xy()))
    plt.clf()
    for pp in patches:
        yhat = savgol_filter(pp[:,1], 51, 3)
        plt.plot(pp[:,0], yhat)

    plt.legend(['Our: positive', 'Our: negative', 'SBM: positive' , 'SBM: negative'], 
                loc="upper left", 
                fontsize=lgndsz,
                frameon=False
                )
    plt.title("Sentiment task", fontsize=titlesz)
    plt.xlabel('\n$y^\mathrm{pred}$\n', fontsize=lblsz, labelpad=lblpd)
    plt.ylabel('\n# of sentences\n', fontsize=lblsz, labelpad=lblpd)
    plt.xticks(fontsize=tksz)
    plt.yticks(fontsize=tksz)
    plt.savefig(pth+'hist_senti.png', bbox_inches='tight')