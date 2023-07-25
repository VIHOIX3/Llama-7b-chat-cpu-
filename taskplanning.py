import os

os.environ['CURL_CA_BUNDLE'] = ''
os.chdir(os.getcwd())

import argparse
import timeit

from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from langchain.llms import CTransformers

# qa_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Context: {context}
# Question: {question}
# Only return the helpful answer below and nothing else.
# Helpful answer:
# """
tp_template1 = """
Please analyze the user request:"{user_requests}",and parse it into one or more tasks, with each task category being assigned from the category list:[text-cls,token-cls,text2text-generation,summarization,translation,question-answering,conversation,text-generation,conversation,tabular-cls]. 
Noting that at least one task category must be resolved, while multiple tasks of either similar or different categories are permissible.
If you can not parse the user request into task, just say that you don't know, don't try to make up an answer.
Only return the task categories below and nothing else.
task categories:
"""
tp_template2 = """
Please analyze the user request:{user_requests},and parse it into one or more tasks, with each task category being assigned from the category list:{task_list}. 
Noting that at least one task category must be resolved, while multiple tasks of either similar or different categories are permissible.
If you can not parse the user request into task, just say that you don't know, don't try to make up an answer.
Only return the task categories below and nothing else.
task categories:
"""
tp_template3 = """
Upon analyzing the user request {user_requests}, it is necessary to parse it into one or more tasks, with each task category being assigned from the designated {task_list}.
Noting that at least one task category must be resolved, while multiple tasks of either similar or different categories are permissible
If you can not parse the user request into task, just say that you don't know, don't try to make up an answer.
Only return the task categories below and nothing else.
task categories:
"""

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='../model/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Location of downloaded GGML model
                    model_type='llama',  # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})


# Wrap prompt template in a PromptTemplate object
def set_tp_prompt(mytemplate):
    prompt = PromptTemplate(template=mytemplate,
                            input_variables=['user_requests', 'task_list'])
    return prompt


def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer()  # Start timer

    user_requests = 'Can you tell me how many people in the text?'
    task_list = 'text-cls,token-cls,text2text-generation,summarization,translation,question-answering,conversation,text-generation,conversation,tabular-cls'
    prompt = set_tp_prompt(tp_template1)

    prompt_text = prompt.format(user_requests=user_requests, task_list=task_list)
    print(prompt_text)

    print(llm(prompt_text))

    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
    #                                    model_kwargs={'device': 'cpu'})
    # vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    # tp = RetrievalQA.from_chain_type(llm=llm,
    #                                  chain_type='stuff',
    #                                  retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
    #                                  return_source_documents=True,
    #                                  chain_type_kwargs={'prompt': prompt})
