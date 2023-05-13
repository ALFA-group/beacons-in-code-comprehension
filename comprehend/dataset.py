import random
import os 
import pandas as pd


def get_dataset_refactory(pth, max_len, use_folders = True, folders = ['1', '2', '3', '4', '5'], number_of_files = 50):
    codes, fixes, identifiers = [], [], []
    if use_folders:
        for folder in folders:
            full_pth = os.path.join(pth, "data", "question_"+folder, 'refactory_online.csv')
            df = pd.read_csv(full_pth)
            
            df_sampled = df.sample(n=number_of_files, random_state=1)
            code = df_sampled['Buggy Code'].tolist()
            codes.extend(code)

            fix = df_sampled['Repair']
            fixes.extend(fix)

            identifier = df_sampled['File Name'].values[0]
            identifiers.extend([identifier])
        
        codes_, fixes_, identifiers_ = [], [], []
        for c, f, i in zip(codes, fixes, identifiers):
            if isinstance(c, str) and isinstance(f, str) and len(c.split(' ')) < max_len:
                codes_.append(c)
                fixes_.append(f)
                identifiers_.append(i)
        
        return fixes_, identifiers_


def get_idx(li, arg):
    return [li[i] for i in arg]


def get_dataset_quixbugs(pth, max_len, number_of_files=50):
    codes, identifiers = [], []
    for f in os.listdir(os.path.join(pth, 'correct_python_programs')):
        if f.endswith('.py'):
            with open(os.path.join(pth, 'correct_python_programs', f), 'r') as fp:
                code = fp.read()
                if len(code.split(' ')) < max_len:
                    codes.append(code)
                    identifiers.append(f)

    idxs = random.sample(range(len(codes)), number_of_files)
    
    return get_idx(codes, idxs), get_idx(identifiers, idxs)


def get_dataset_custom(pth, max_len, number_of_files=50):
    df = pd.read_excel(os.path.join(pth, 'codes.xlsx'), index_col=None)
    codes = df['fn_def'].tolist()
    idxs = list(range(len(codes)))
    if number_of_files == -1:
        return codes, idxs
    else:
        return codes[:number_of_files], idxs[:number_of_files]

def get_dataset_custom_anonym(pth, max_len, number_of_files=50):
    df = pd.read_excel(os.path.join(pth, 'codes.xlsx'), index_col=None)
    codes = df['anonymous_fn_def'].tolist()
    idxs = list(range(len(codes)))
    if number_of_files == -1:
        return codes, idxs
    else:
        return codes[:number_of_files], idxs[:number_of_files]

def get_dataset_fn(datasetname):
    if datasetname == 'refactory':
        return get_dataset_refactory
    elif datasetname == 'quixbugs':
        return get_dataset_quixbugs
    elif datasetname == 'custom':
        return get_dataset_custom
    elif datasetname == 'custom-anonym':
        return get_dataset_custom_anonym
