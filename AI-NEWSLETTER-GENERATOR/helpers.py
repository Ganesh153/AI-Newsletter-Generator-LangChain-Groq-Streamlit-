import logging
import os, json
from langchain_groq import ChatGroq
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
serp_api_key=os.getenv("SERPER_API_KEY")

embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Serp request to get the list of relevant articles related to query.
def serp_req(query):
    search=GoogleSerperAPIWrapper(k=5,type="search")
    response=search.results(query)
    logging.info(f"Search Results: {response}")
    return response

# LLM to choose the best articles and return urls
def urls_picker(response,query):
    # turn json to str
    response_str=json.dumps(response)

    llm=ChatGroq(model="openai/gpt-oss-120b",api_key=groq_api_key,temperature=0.7)

    template="""
    You are a world class journalist, researcher, tech, software Engineer, Developer and a online course creator
    , you are amazing at finding the most interesting and relevant, useful articles in certain topics.

    Response:{response_str}

    Above is the list of search results for the query: {query}

    Please choose the best 3 articles from the list and return ONLY an array of the urls.
    Do not include anything else-
    return ONLY an array of the urls.
    Also make sure the articles are recent and not too old.
    If the file, or URL is invalid, show www.google.com.
"""
    prompt_template=PromptTemplate(
        input_variables=['response_str','query'],
        template=template
    )

    urls_chain=LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )

    urls=urls_chain.run(response_str=response_str,query=query)

    #convert string to list
    res_urls=json.loads(urls)

    return res_urls

def extract_content(urls,persist_dir='faiss_index'):
    # if os.path.exists(persist_dir):
    #     print("Existing database found...")
    #     db=FAISS.load_local(persist_dir,embeddings)
    #     return db
    
    loader=UnstructuredURLLoader(urls)
    url_load=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs=text_splitter.split_documents(url_load)

    db=FAISS.from_documents(docs,embeddings)

    # db.save_local(persist_dir)
    return db

def summarizer(db,query):
    docs=db.similarity_search(query,k=4)

    doc_page_content=" ".join([d.page_content for d in docs])

    llm=ChatGroq(model="openai/gpt-oss-120b",api_key=groq_api_key,temperature=0.7)

    template="""
    {docs}
    As a world class journalist, researcher, article, newsletter and blog writer,
    you will summarize the text above in order to create a 
    newsletter around {query}.
    This newsletter will be sent as an email. The format is going to be like
    Tim Ferriss' "5-Bullet Friday" newsletter.

    Please follow all of the following guidelines:
    1. Make sure the content is engaging, informative with good data.
    2. Make sure the content is not too long, it should be the size of a nice newsletter.
    3. The content should address the {query} topic very well.
    4. The content needs to be good and informative
    5. The content needs to be written in a way that is easy to read, digest and understandable.
    6. The content needs to give the audience actinable advice and insights.

    Summary:

    """
    prompt_template=PromptTemplate(input_variables=["docs","query"],template=template)

    summarizer_chain=LLMChain(llm=llm, prompt=prompt_template,verbose=True)

    response=summarizer_chain.run(docs=doc_page_content,query=query)

    return response.replace("\n","")

def generate_newsletter(summary,query):
    summaries_str=str(summary)
    llm=ChatGroq(model="openai/gpt-oss-120b",api_key=groq_api_key,temperature=0.7)
    template="""
    {summaries_str}
    As a world class journalist, researcher, article, newsletter and blog writer,
    you will use the text above as the context about {query}
    to write an excellent newsletter to be sent to subscribers.

    This newsletter will be sent as mail as well. Format is going to be like Tim Ferriss' "5-Bullet Friday" newsletter.

    Make sure to write it informally - no "Dear" or any other formal words. 
    Just follow format like:
    `Hi All!
        Here is your weekly dose of Newsletter.......`
    Make sure to also write a backstory about the topic - make it personal, engaging, 
    going to the meat of the newsletter.

    Follow these Guidelines:
    1. Make sure the content is engaging, informative with good data.
    2. Make sure the content is not too long, it should be the size of a nice newsletter.
    3. The content should address the {query} topic very well.
    4. Content needs to be good and informative to all who are reading that newsletter.
    5. Content needs to be written in a way that is easy to read, digest and engaging.
    6. Content needs to give the audience actionable advice and insights.

    If there are book, or products involved, make sure to add links.

    As a signoff, write a clever quote related to the learning, general wisdom.

    Sign with "P.T.Oss
            - Learner and Writer"

    NewsLetter--->
"""
    prompt_template=PromptTemplate(input_variables=['summaries_str','query'],
                                   template=template)
    generator_chain=LLMChain(llm=llm, prompt=prompt_template,verbose=True)

    response=generator_chain.predict(summaries_str=summaries_str, query=query)

    return response