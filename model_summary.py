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
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import numpy as np
import whisper
import torch
from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram

def extract_video_id(url):
    video_id = url.split("v=")[1]
    if "&" in video_id:
        video_id = video_id.split("&")[0]
    return video_id

def download_youtube_audio(url):
    # Create the "tmp" directory if it doesn't exist
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    yt = YouTube(url)
    video_id = extract_video_id(url)

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
    # file_size = os.path.getsize(file_path)
    # size_mb = file_size / (1024 * 1024)

<<<<<<< HEAD
    # Check if the file size is less than 20 MB
    # if size_mb < 20:
    with open(file_path, "rb") as audio_file:
        # Transcribe the audio using OpenAI API
        # model = whisper.load_model("large-v3")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        # model = whisper.load_model("base")
        # client = OpenAI()
        # result = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file, response_format="text") 
        transcript_text = transcript["text"]
        with open(transcript_filepath, "w") as transcript_file:
            transcript_file.write(transcript_text)
=======
    # Check if the file size is less than 25 MB
    if size_mb < 25:
        with open(file_path, "rb") as audio_file:
            # Transcribe the audio using OpenAI API
            transcript = openai.Audio.transcribe(file=audio_file, model="whisper-1", response_format="text", language="en")
            with open(transcript_filepath, "w") as transcript_file:
                transcript_file.write(transcript)
>>>>>>> 580b8f0a365c5ed3da6842a7e1256dc4393c636b

    # Delete the audio file
    os.remove(file_path)

    # else:
    #     print("Size too large, please provide audio file with size <20 MB.")
  

@st.cache_data(show_spinner=False)
def generate_video_summary(api_key, url):
    openai.api_key = api_key
<<<<<<< HEAD
    llm = OpenAI(temperature=0.5, openai_api_key=api_key, model_name="gpt-3.5-turbo-1106")
=======
    llm = OpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo-16k")
>>>>>>> 580b8f0a365c5ed3da6842a7e1256dc4393c636b
    text_splitter = CharacterTextSplitter()

    video_id = extract_video_id(url)


    # The path of the audio file and transcript
    audio_path = f"tmp/{video_id}.mp3"
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exists
    if os.path.exists(transcript_filepath):
        with open(transcript_filepath) as f:
            transcript_file = f.read()
        
    else:
        download_youtube_audio(url)
        transcribe_audio(audio_path, video_id)

        with open(transcript_filepath) as f:
            transcript_file = f.read()

    texts = text_splitter.split_text(transcript_file)
<<<<<<< HEAD
    documents = [Document(page_content=t) for t in texts[:3]]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(documents)

=======
    docs = [Document(page_content=t) for t in texts]    
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
>>>>>>> 580b8f0a365c5ed3da6842a7e1256dc4393c636b
    return summary.strip()

def generate_answer(api_key, url, question):
    openai.api_key = api_key
<<<<<<< HEAD
    llm = OpenAI(temperature=0.5, openai_api_key=api_key, model_name="gpt-3.5-turbo-1106")
    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=25)
=======
    llm = OpenAI(temperature=0, openai_api_key=api_key, model_name="gpt-3.5-turbo-16k")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
>>>>>>> 580b8f0a365c5ed3da6842a7e1256dc4393c636b

    video_id = extract_video_id(url)
    transcript_filepath = f"tmp/{video_id}.txt"

    # Check if the transcript file already exists
    if os.path.exists(transcript_filepath):
        loader = TextLoader(transcript_filepath, encoding='utf8')
        documents = loader.load()
        

    else: 
        download_youtube_audio(url)
        audio_path = f"tmp/{video_id}.mp3"
        
        # Transcribe the mp3 audio to text and Generating summary
        transcribe_audio(audio_path, video_id)
        loader = TextLoader(transcript_filepath, encoding='utf8')
        documents = loader.load()

    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(deployment="text-embedding-ada-002")
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.run(question)

    return answer.strip()
