import os

os.environ['CURL_CA_BUNDLE'] = ''
os.chdir(os.getcwd())

os.environ["OPENAI_API_KEY"] = "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"

import timeit

from langchain import PromptTemplate

from langchain import OpenAI
from langchain.llms import CTransformers

tp_template0 = """Use the following pieces of information to answer the category of the context, with the category being assigned from the category list: [{task_list}].
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {user_requests}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

tp_template1 = """
Please analyze the task content:"{user_requests}", and tell me the task category, with the category being assigned from the category list: [{task_list}]. 
Noting that only one task category can be chosen, if you can not parse the user request into task, just say that you don't know, don't try to make up an answer.
Only return helpful category below and nothing else.
Helpful task category:
"""

tp_template3 = """
Base on the model {model} you parse and the context {context}, select one category which belongs to the category list {task_list}.

Noting that at least one task category must be resolved, while multiple tasks of either similar or different categories are permissible
If you can not parse the user request into task, just say that you don't know, don't try to make up an answer.
Only return the task categories below and nothing else.
helpful task categories:
"""

# Local CTransformers wrapper for Llama-2-7B-Chat
llama2 = CTransformers(model='./model/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Location of downloaded GGML model
                       model_type='llama',  # Model type Llama
                       config={'max_new_tokens': 256,
                               'temperature': 0.01})


# gpt = OpenAI(temperature=0)


# Wrap prompt template in a PromptTemplate object
def set_tp_prompt(mytemplate):
    prompt = PromptTemplate(template=mytemplate,
                            input_variables=['user_requests', 'task_list'])
    return prompt


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input', type=str)
    # args = parser.parse_args()
    start = timeit.default_timer()  # Start timer

    user_requests1 = 'please generate a sentence which includes COVID-19 is transmitted via droplets and air-borne'
    user_requests2 = 'please summarize the text and tell me what it talks about '
    user_requests3 = 'give me a similar image with example image'
    user_requests4 = 'what does the video talk about?'
    task_list = 'text-cls,token-cls,text2text-generation,summarization,translation,question-answering,conversation,text-generation,conversation,tabular-cls,image-to-image'
    model_list = 'text,image,video,audio'
    prompt = set_tp_prompt(tp_template0)

    # select model
    prompt_text1 = prompt.format(user_requests=user_requests1, task_list=model_list)
    prompt_text2 = prompt.format(user_requests=user_requests2, task_list=model_list)
    prompt_text3 = prompt.format(user_requests=user_requests3, task_list=model_list)
    prompt_text4 = prompt.format(user_requests=user_requests4, task_list=model_list)

    # 具体任务识别有问题
    # prompt_text1 = prompt.format(user_requests=user_requests1, task_list=task_list)
    # prompt_text2 = prompt.format(user_requests=user_requests2, task_list=task_list)
    # prompt_text3 = prompt.format(user_requests=user_requests3, task_list=task_list)
    # prompt_text4 = prompt.format(user_requests=user_requests4, task_list=task_list)
    # print(prompt_text)
    times = 1
    for i in range(times):
        print('=' * 50)  # Formatting separator
        print("Llama2:\n ", llama2(prompt_text1))
        print('=' * 50)
        print(llama2(prompt_text2))
        print('=' * 50)
        print(llama2(prompt_text3))
        print('=' * 50)
        print(llama2(prompt_text4))
        # print("Chatpt3: ", gpt(prompt_text1), gpt(prompt_text2))
        # print('Llama2: ',llama2(prompt_text4))

    end = timeit.default_timer()  # End timer
    print(f"Time to response: {(end - start)}")
