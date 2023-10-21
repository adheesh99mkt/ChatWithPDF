import os
import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

#sidebar content
with st.sidebar:
    st.title("LLM chatApp")
    st.markdown('''
        ## About
        This is an LLM chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [Langchain](https://python.langchain.com)
        - [OpenAi](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space(5)
    st.write("made by ADHEESH K V")

def main():
      st.header("Chat with PDF")

      load_dotenv()

      #upload a PDf file
      pdf=st.file_uploader("upload your PDF here",type='pdf')
      if pdf is not None:
        pdf_reader=PdfReader(pdf)
        
        text=""
        for page in pdf_reader.pages:
             text += page.extract_text()
        
        text_splitter=RecursiveCharacterTextSplitter(
             chunk_size=1000,
             chunk_overlap=200,
             length_function=len
            )
        chunks=text_splitter.split_text(text=text)
        #embeddings
        embeddings=OpenAIEmbeddings()
        VectorStore=FAISS.from_texts(chunks, embedding=embeddings)
        store_name=pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore=pickle.load(f)
            #st.write("Embeddings loaded from the disc")
        else:
            embeddings=OpenAIEmbeddings()
            VectorStore=FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(VectorStore,f)
            #st.write("Embeddings computations are completed")
        #accept user question/query

        query=st.text_input("Ask question about your pdf file:")
        #st.write(query)
        if query:
            docs=VectorStore.similarity_search(query=query,k=3)

            llm=OpenAI(model_name='gpt-3.5-turbo')
            chain=load_qa_chain(llm=llm,chain_type="stuff")
            with get_openai_callback() as cb:
                response=chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)
        


if __name__=='__main__':
        main()