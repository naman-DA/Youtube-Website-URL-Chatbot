import streamlit as st
import validators
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (YoutubeLoader,UnstructuredURLLoader,)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api._errors import RequestBlocked

# Streamlit Config

st.set_page_config(page_title = "LangChain: Summarize URL", layout = "centered",)
st.title("LangChain : Summarize text from YouTube or Website URL")
st.subheader("Summarize URL")

# API Key (Secrets + Optional Override)
groq_api_key = st.secrets.get("GROQ_API_KEY", "")

with st.sidebar:
    st.markdown("### Groq API Key")
    user_key = st.text_input("Optional override", type="password")
    
    if user_key:
        groq_api_key = user_key

if not groq_api_key:
    st.warning("Please add your Groq API key in Streamlit Secrets.")
    st.stop()

# URL Input
generic_url = st.text_input("URL", label_visibility="collapsed")

# LLM

llm = ChatGroq(model_name = "llama-3.1-8b-instant", groq_api_key = groq_api_key,)

prompt = PromptTemplate(template = """
Provide a clear and concise summary (maximum 400 words)
of the following content:
{text}
""", input_variables=["text"],)

if st.button("Summarize the content from YT or Website"):
    if not generic_url.strip():
        st.error("Please provide a URL.")
        st.stop()

    if not validators.url(generic_url):
        st.error("Please enter a valid URL.")
        st.stop()

    is_youtube = "youtube.com" in generic_url or "youtu.be" in generic_url

    try:
        with st.spinner("Fetching content..."):
          if is_youtube:
            try:
              loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info = False, language = ["en"],)
              docs = loader.load()
            except RequestBlocked:
              st.error("YouTube blocked transcript access from this server.")
              st.info(
                      "This is a known limitation on Streamlit Cloud.\n\n"
                      "You can:\n"
                      "• Try another YouTube video\n"
                      "• Use a website URL\n"
                      "• Or add Whisper-based transcription (advanced)"
              )
              st.stop()
          else:
            loader = UnstructuredURLLoader(urls = [generic_url], ssl_verify = False, headers = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},)
            docs = loader.load()
            
        if not docs:
            if is_youtube:
                st.error("No transcript found.")
                st.info(
                    "This YouTube video does not have captions enabled. "
                    "Please try another video with subtitles."
                )
            else:
                st.error("No readable content found on this webpage.")
            st.stop()

        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())

        if not text.strip():
            if is_youtube:
                st.error("No transcript found.")
                st.info(
                    "This YouTube video does not have captions enabled. "
                    "Please try another video with subtitles."
                )
            else:
                st.error("No readable text found on this webpage.")
                st.info("This website may block scraping or use dynamic content.")
            st.stop()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200,)
        chunks = splitter.split_text(text)

        if not chunks:
            st.error("Failed to split content.")
            st.stop()

        if len(chunks) > 20:
            st.warning("Content is very long. Summarizing the first part only.")

        # Summarization

        with st.spinner("Summarizing..."):
            chain = prompt | llm | StrOutputParser()
            summaries = [
                chain.invoke({"text": chunk})
                for chunk in chunks[:5]
            ]

        st.success("\n\n".join(summaries))
    except Exception as e:
      if "RequestBlocked" in str(e):
        st.error("YouTube blocked the request. Try running locally or using a proxy.")
      else:
        st.error("Something went wrong.")
        st.exception(e)
