import openai
from openai.error import RateLimitError
import logging
import os

'''
    An interface to OpenAI models to predict text and provide an interactive chat interface.
    Supported models: gpt3-davinci (davinci), ChatGPT (chat), GPT-4 (chat-gpt4)

    Author: Stephen Moskal (smoskal@csail.mit.edu)
    MIT CSAIL ALFA Group
'''


class OpenAIInterface:
    def __init__(self, api_key=None):
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")

        if not openai.api_key:
            logging.getLogger('sys').warning(f'[WARN] OpenAIInterface (Init): No OpenAI API key given!')

    def predict_text(self, prompt, max_tokens=100, temp=0.5, mode='chat', prompt_as_chat=False):
        '''
        Queries OpenAI's GPT-3 model given the prompt and returns the prediction.
        See: openai.Completion.create()
            engine="text-davinci-002"
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0

        @param prompt: Prompt describing what you would like out of the
        @param max_tokens: Length of the response
        @return:
        '''

        try:
            if mode == 'chat':
                if prompt_as_chat:
                    message = prompt
                else:
                    message = [{"role": "user", "content": prompt}]

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=message,
                    temperature=temp
                )
                return response['choices'][0]['message']['content']
            elif mode == 'chat-gpt4':
                if prompt_as_chat:
                    message = prompt
                else:
                    message = [{"role": "user", "content": prompt}]

                response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=message,
                temperature=temp
                )
                return response['choices'][0]['message']['content']
            else:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=temp,
                    max_tokens=max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                return response.choices[0].text
        except RateLimitError as e:
            logging.getLogger('sys').error(f'[WARN] OpenAIInterface: Rate Limit Exceeded! {e}')
            return '0'
        except Exception as e:
            logging.getLogger('sys').error(f'[ERROR] OpenAIInterface: Unexpected exception- {e}')
            return '-1'

    def start_interactive_chat(self, chat_model='chat', max_response_tokens=350):
        chat_history = []

        print("Input your system prompt (q to quit):")
        system_prompt_text = input()
        if system_prompt_text == ('q' or 'quit'):
            print('Aborting chat mode...')
            return
        system_prompt = {'role': 'system', 'content': system_prompt_text}
        chat_history.append(system_prompt)

        while True:
            print('Prompt the model (q to quit, h for history, r to reset history):')
            prompt_text = input()

            if prompt_text == ('q' or 'quit'):
                print('Aborting chat mode...')
                return
            elif prompt_text == ('h' or 'history'):
                print(chat_history)
            elif prompt_text == ('r' or 'reset'):
                print('Resetting the chat history')
                chat_history = [system_prompt]
            else:
                try:
                    prompt_format = {'role': 'user', 'content': prompt_text}
                    chat_history.append(prompt_format)
                    chat_response = self.predict_text(chat_history, max_tokens=max_response_tokens, mode=chat_model,
                                                        prompt_as_chat=True)
                    print(f'Response: {chat_response}\n')
                    chat_history.append({'role': 'assistant', 'content': chat_response})
                except Exception as e:
                    print(f'Unknown exception: {e}')
                    print('Do you want to continue? (y/n)')
                    selection = input()
                    if selection == 'n':
                        print('Aborting chat mode...')
                        return

if __name__ == '__main__' :
    personal_api_key = open('openai_api_key_sam.key', 'r').read().strip()
    openai_interface = OpenAIInterface(api_key=personal_api_key)
    print(openai_interface.predict_text('I am a dog. What are you?', mode='chat-gpt4'))