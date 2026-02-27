import streamlit as st
import validators
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter

## Stramlit app

st.set_page_config(page_title = "Langchain : Summarize text from Youtube or Website URL")
st.title("Langchain : Summarize text from Youtube or Website URL")
st.subheader("Summarize URL")

## Get the Groq api key & URL(Youtube or Website) to be summarized

groq_api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("### Groq API Key")
    user_key = st.text_input("Optional override", type="password")
    
    if user_key:
        groq_api_key = user_key

if not groq_api_key:
    st.warning("Please add your Groq API key in Streamlit Secrets.")
    st.stop()

generic_url = st.text_input("URL", label_visibility = "collapsed")
llm = ChatGroq(model_name="llama-3.1-8b-instant", groq_api_key = groq_api_key)

prompt_template = """
Provide a summary of the following content in 400 words:
Summary:{text}
"""

prompt = PromptTemplate(template = prompt_template, input_variables = ["text"])

if st.button("Summarize the content from YT or Website"):
  
  ## Validate inputs
  
  if not groq_api_key.strip() or not generic_url.strip():
    st.error("Please provide the info")
  elif not validators.url(generic_url):
    st.error("Please enter the valid URL")
  else:
    try:
      with st.spinner("Summarizing...."):
        
        ## Loading the YT or Website video content
        
        if "youtube.com" in generic_url:
          loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = False)
        else:
          loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify = False, 
                    headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                  )
        docs = loader.load()

      if not docs:
        st.error("No readable content or transcript found.")
        st.stop()
        
      text = "\n\n".join([doc.page_content for doc in docs if doc.page_content.strip()])
        
      if not text.strip():
        st.error("No transcript found.")
        st.info("This video does not have captions." "Please try another video with subtitles enabled.")
        st.stop()

      text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200)
      chunks = text_splitter.split_text(text)

      if len(chunks) > 20:
        st.warning("Video is too long. Summarizing first 20 minutes only.")
    
      ## Chain for Summarization
          
      chain = prompt | llm | StrOutputParser()
      summaries = []
        
      for chunk in chunks[:5]:
        summaries.append(chain.invoke({"text": chunk}))

      st.success("\n".join(summaries))   
    except Exception as e:
      st.exception(f"Exception{e}")