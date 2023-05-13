from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from transformers import RobertaTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from transformers import pipeline
from transformers import RobertaTokenizer
from transformers import PLBartTokenizer, PLBartForConditionalGeneration


def get_models(models):
    model_and_tokenizer = []
    for m in models:
        if m == 'plbart':
            checkpoint = 'uclanlp/plbart-base'
        
        if m == 'codeberta-small':
            checkpoint = 'huggingface/CodeBERTa-small-v1'
        
        if m == 'codeberta-base-mlm':
            checkpoint = "microsoft/codebert-base-mlm"
        
        if m == "gpt-neo":
            checkpoint = "EleutherAI/gpt-neo-125M"
        
        if m == 'codet5':
            checkpoint = 'Salesforce/codet5-base'

        if m == 'santa-coder':
            checkpoint = "bigcode/santacoder"
        
        if m == 'gpt-neox':
            checkpoint = "EleutherAI/gpt-neox-20b"
        
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True)
        model_and_tokenizer.append((model, tokenizer, config))

        #tokenizer = GPT2Tokenizer.from_pretrained("shibing624/code-autocomplete-distilgpt2-python")
        #model = GPT2LMHeadModel.from_pretrained("shibing624/code-autocomplete-distilgpt2-python")

        #tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-neo-125M-code-clippy-code-search-py")
        #model = AutoModelForCausalLM.from_pretrained("flax-community/gpt-neo-125M-code-clippy-code-search-py")

        #tokenizer = AutoTokenizer.from_pretrained("lvwerra/codeparrot")
        #model = AutoModelForCausalLM.from_pretrained("lvwerra/codeparrot")

        #tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
        #model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

        #tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
        #model = AutoModelForMaskedLM.from_pretrained("huggingface/CodeBERTa-small-v1")

        #model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
        #tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
    return model_and_tokenizer
