# import streamlit as st
# import os
# import streamlit as st
# from dotenv import load_dotenv
from pytube.exceptions import VideoUnavailable
# from moviepy.editor import *
# from pytube import YouTube
import langchain as lc
import tempfile
import openai
import os 
import openai
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
from moviepy.editor import *
from moviepy.editor import AudioFileClip
from pytube import YouTube
from pydub import *
import tempfile
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# from model_summary import generate_answer, generate_video_summary

# load_dotenv()

headers = {
    "auhorization": st.secrets['OPENAI_API_KEY'],
    "COntent-type": "Applicaion/json"
}

# st.write("api_key", st.secrets["OPENAI_API_KEY"])

# st.write(
# 	"Has environment variables been set:",
# 	os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"])

openai_api_key = os.getenv("OPENAI_API_KEY")


def valid_url(url: str) -> bool:
    try:
        yt = YouTube(url)
        if not yt.video_id:
            return False
    except (VideoUnavailable, Exception):
        return False
    return yt.streams.filter(adaptive=True).first() is not None

def video_info(url: str):
    yt = YouTube(url)
    title = yt.title
    return title
def download_audio(url: str):
    # Create the "tmp" directory if it doesn't exist
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    yt = YouTube(url)

    # Extract the video_id from the URL
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]

    # Get the first available audio stream and download it
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path=tmp_dir)

    # Convert the downloaded audio file to MP3 format
    audio_path = os.path.join(tmp_dir, audio_stream.default_filename)
    audio_clip = AudioFileClip(audio_path)
    audio_clip.write_audiofile(os.path.join(tmp_dir, f"{video_id}.mp3"))

    # Delete the original audio stream
    os.remove(audio_path)

def transcribe_audio(file_path, video_id):
    # The path of the transcript
    transcript_filepath = f"tmp/{video_id}.txt"
    
    # Get the size of the file in bytes and convert ot megabytes
    file_size = os.path.getsize(file_path)
    file_size_in_mb = file_size / (1024 * 1024)

    # Check if the file size is less than 25 MB
    if file_size_in_mb < 25:
        with open(file_path, "rb") as audio_file:
            # Transcribe the audio using OpenAI API
            transcript_result = openai.Audio.transcribe("whisper-1", audio_file)
            transcript_text = transcript_result["text"]
            with open(transcript_filepath, "w") as transcript_file:
                transcript_file.write(transcript_text)

        # Delete the audio file
        os.remove(file_path)

    else:
        print("Size too large, please provide audio file with size <25 MB.")
@st.cache_data(show_spinner=False)
def generate_video_summary(api_key: str, url: str) -> str:
    openai.api_key = api_key
    llm = OpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo")
    text_splitter = CharacterTextSplitter()

    # Extract the video_id from the URL
    query_info = urlparse(url).query
    params_info = parse_qs(query_info)
    video_id = params_info["v"][0]

    # The path of the audio file and transcript
    audio_path = f"tmp/{video_id}.mp3"
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exists
    if os.path.exists(transcript_filepath):
        with open(transcript_filepath) as f:
            transcript_file = f.read()

    else:
        download_audio(url)
        transcribe_audio(audio_path, video_id)

        with open(transcript_filepath) as f:
            transcript_file = f.read()

    texts = text_splitter.split_text(transcript_file)
    docs = [Document(page_content=t) for t in texts[:3]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)

    return summary.strip()

def generate_answer(api_key: str, url: str, question: str) -> str:
    llm = OpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo")
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=25)

    # Extract the video_id from the url
    query = urlparse(url).query
    params = parse_qs(query)
    video_id = params["v"][0]
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exists
    if os.path.exists(transcript_filepath):
        loader = TextLoader(transcript_filepath, encoding='utf8')
        documents = loader.load()
    else: 
        download_audio(url)
        audio_path = f"tmp/{video_id}.mp3"
        
        # Transcribe the mp3 audio to text and Generating summary
        transcribe_audio(audio_path, video_id)
        loader = TextLoader(transcript_filepath, encoding='utf8')
        documents = loader.load()

    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)

    return answer.strip()

st.set_page_config(page_title="YouTubeGPT")

#  main app UI
def main():
    with st.sidebar:
        st.markdown("""<div style="text-align: justify;">Introducing YouTube Summarizer app powered by OpenAI, Langchain and Streamlit! 🚀
        Now, watching videos becomes hassle-free. Simply paste the YouTube link, and voila!
        Get instant, concise summaries of your favorite content and feel free to ask questions for more insight. Effortless, efficient, and tailored for you.
        Enhancing your video experience to the next level!
        </div>""", unsafe_allow_html=True)
        st.markdown("#")
        st.markdown("Source code [GitHub](https://github.com/YohansAlemayehu?tab=repositories)")
        st.markdown("LinkedIn [Yohans Brhane](https://www.linkedin.com/in/yohans-brhane-46473063/)")
    st.header(":orange[YouTubeGPT Video Summarizer]")
    st.subheader(":orange[Discover video insights. Built with OpenAI & Langchain] 🚀")
    st.markdown('#') 
    choice = st.radio("Go ahead and make your selection:", ('Video Summary', 'Question-Answering'), horizontal=True)
    st.markdown('#') 

    # os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

    # openai_api_key = os.getenv('OPENAI_API_KEY')

    # Enter yourtube URL
    youtube_url = st.text_input("Enter YouTube Video URL")

    
    if valid_url(youtube_url):
        video_title = video_info(youtube_url)
        st.markdown(f"##### {video_title}")
        st.video(youtube_url)
        os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
        openai_api_key = os.getenv('OPENAI_API_KEY')
    else:
        st.error("Please enter a valid YouTube URL.")

    if choice == "Video Summary":
        if st.button("Summary"):
            if not valid_url:
                st.warning("Please enter a valid YouTube URL.")
            else:
                with st.spinner("Generating summary..."):
                    summary = generate_video_summary(openai_api_key, youtube_url)
                    st.markdown(f"##### Summary of the Video:")
                    st.success(summary)

    elif choice == "Question-Answering":
        if valid_url:
            st.markdown('##### Would you like to explore further details regarding this video?')
            question = st.text_input("Submit your questions here.")
        else:
            st.markdown('##### Would you like to explore further details regarding this video?')
            question = st.text_input("Submit your questions here.", disabled=True)
            
        if st.button("Answer"):
            if not valid_url:
                st.warning("Please enter a valid YouTube URL.")
            elif not question:
                st.warning("Please enter your question.")
            else:
                with st.spinner("Generating answer..."):
                    answer = generate_answer(openai_api_key, youtube_url, question)
                st.success(answer)

if __name__ == "__main__":
    main()