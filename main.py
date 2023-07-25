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

qa_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='../model/llama-2-7b-chat.ggmlv3.q8_0.bin',  # Location of downloaded GGML model
                    model_type='llama',  # Model type Llama
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})


# Wrap prompt template in a PromptTemplate object
def set_qa_prompt():
    prompt = PromptTemplate(template=qa_template,
                            input_variables=['context', 'question'])
    return prompt


# Build RetrievalQA object
def build_retrieval_qa(llm, prompt, vectordb):
    dbqa = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt})
    return dbqa


# Instantiate QA object
def setup_dbqa():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    vectordb = FAISS.load_local('vectorstore/db_faiss', embeddings)
    qa_prompt = set_qa_prompt()
    dbqa = build_retrieval_qa(llm, qa_prompt, vectordb)

    return dbqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args = parser.parse_args()
    start = timeit.default_timer()  # Start timer

    # Setup QA object
    dbqa = setup_dbqa()

    # Parse input from argparse into QA object
    response = dbqa({'query': args.input})
    end = timeit.default_timer()  # End timer

    # Print document QA response
    print(f'\nAnswer: {response["result"]}')
    print('=' * 50)  # Formatting separator

    # Process source documents for better display
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i + 1}\n')
        print(f'Source Text: {doc.page_content}')
        print(f'Document Name: {doc.metadata["source"]}')
        print(f'Page Number: {doc.metadata["page"]}\n')
        print('=' * 50)  # Formatting separator

    # Display time taken for CPU inference
    print(f"Time to retrieve response: {end - start}")
