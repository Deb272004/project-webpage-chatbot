import re
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings#
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings



import os

load_dotenv()

class WebSearch:
    
    def __init__(self, persist_path="vectorstore_index"):
        print("welcome to the chatbot how can i help you")
        self.api_key = os.getenv("groq_api")
        self.google_api_key = os.getenv("gemini_api")
        self.hugging_face_key = os.getenv("HUGGING_FACE_HUB_API_TOKEN")
        self.__chathistory = [SystemMessage(content=f"You are an helpful assistant,answer the user query like an knowledge guide from the provided")]
        self.threshold = 1.0
        self.persist_path = persist_path
        self.extracted_query = "",
        self.vectorstore = self._load_vectorstore()
    
    def _load_vectorstore(self):
        if os.path.exists(self.persist_path):
            print("loading info from existing vector store")
            # embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
            # embedding = OllamaEmbeddings(model='mistral')
            
            embedding = HuggingFaceEndpointEmbeddings(model="jinaai/jina-embeddings-v4", huggingfacehub_api_token=self.hugging_face_key)
            
            return FAISS.load_local(self.persist_path, embedding, allow_dangerous_deserialization=True)
        print("no info found")
        return None
        
    def _save_vectorstore(self):
        if self.vectorstore:
            os.makedirs(self.persist_path, exist_ok=True)
            self.vectorstore.save_local(self.persist_path)
            print(f"info saved at {self.persist_path}")


    def answer_qeury(self, query):
        self.extracted_query,link= self.extract_link_query(query)
        print("query",self.extracted_query)
        print("link",link)
        
        if link != None:
            if not self.url_already_in_store(link):
                retrieved_content = self.document_laod(link)
                if retrieved_content != None:
                    print("answering from vector store")
                    self.llm_conn(self.extracted_query,retrieved_content)
                else:
                    print("answering from duckducgo 1")
                    self.duckduckgo_function(self.extracted_query)
            else:
                retrieved_text_from_vec= self.retriever_and_score(self.extracted_query)
                print("answering from vector store")
                self.llm_conn(self.extracted_query, retrieved_text_from_vec)
        else:
            retrieved_text_from_query = self.retriever_and_score(self.extracted_query)
            if retrieved_text_from_query != None:
                print("answering from vector store")
                self.llm_conn(self.extracted_query, retrieved_text_from_query)
            else:
                print("answering from duckducgo 2")
                self.duckduckgo_function(self.extracted_query)
            
    
    def extract_link_query(self,query):
        match = re.search(r'(https?://\S+|www\.\S+)', query)
        url = match.group(0) if match else None
        query = re.sub(r'(https?://\S+|www\.\S+)', '', query).strip()
        return query, url
    
    def duckduckgo_function(self, query):
        context = self.duckduckgo(query)
        self.llm_conn(query,context)
    
    def duckduckgo(self, query):
        search = DuckDuckGoSearchRun()
        response = search.invoke(query)
        return response

    def llm_conn(self, query, context):

        model = ChatGroq(model="llama-3.1-8b-instant",api_key = self.api_key)
        self.__chathistory.append(HumanMessage(content=f"answer the {query} from the following {context}"))
        result = model.invoke(self.__chathistory)
        self.__chathistory.append(AIMessage(content=result.content))
        print(result.content)

    def retriever_and_score(self, query):
        if not self.vectorstore:
          return None
        filtered_content = []
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=5)
        for doc, score in docs_and_scores:
            if score <= self.threshold:
                filtered_content.append(doc)
        
        return filtered_content if filtered_content else None

    
    def url_already_in_store(self, link):
        if self.vectorstore is None:
            return False
        results = self.vectorstore.similarity_search(link, k=1)
        return any(link == doc.metadata.get("source") for doc in results)
        
    
    def document_laod(self,link):
        loader = RecursiveUrlLoader(
            url=link,
            max_depth=1,
            extractor= self.bs4_extractor,
            prevent_outside=True
        )
        docs = list(loader.lazy_load())
        splitted_text = self.text_splitter(docs,link) 
        self.create_vector_store(splitted_text)

        content = self.retriever_and_score(self.extracted_query)
        return content


    def bs4_extractor(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    def text_splitter(self, docs, link):
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        texts = [doc.page_content for doc in docs]
        splitted_docs = text_splitter.create_documents(texts, metadatas=[{"source": link}] * len(texts))

        return splitted_docs
    
    def create_vector_store(self, splitted_docs):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.google_api_key)
        if self.vectorstore:
            self.vectorstore.add_documents(splitted_docs)
        else:
            self.vectorstore = FAISS.from_documents(splitted_docs, embeddings)
        self._save_vectorstore()



web = WebSearch()

web.answer_qeury("https://www.geeksforgeeks.org/machine-learning/k-nearest-neighbours/")


