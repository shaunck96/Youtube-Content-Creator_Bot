import pandas as pd
import scrapetube
from pytube import YouTube
import os
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os
import openai
import regex as re
import json
import requests
from tqdm import tqdm
from googleapiclient.discovery import build
import regex as re
from transformers import pipeline
from faster_whisper import WhisperModel
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from faster_whisper import WhisperModel
import tiktoken
import pandas as pd
import re
from urllib import parse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import ast
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from itertools import islice
from youtube_comment_downloader import *
from .utils import *

# Open the text file for reading
file_path = './sniper_wolf.txt'  # Replace 'urls.txt' with the path to your text file
url_list = []

try:
    with open(file_path, 'r') as file:
        # Read lines from the file and strip whitespace
        for line in file:
            url = line.strip()
            url_list.append(url)
except FileNotFoundError:
    print(f"The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

model_size = "small"#"large-v2"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

transcriptions = {}

for path,url in url_mp3_mapping.items():
  transcription = transcriber(path,model)
  video_id = YouTube(url).video_id
  transcriptions[video_id] = transcription

formatted_transcriptions = []

for key, transcription in transcriptions.items():
    formatted_transcription = f"Transcription {key} \n"
    formatted_transcription += '\n'.join(transcription)
    formatted_transcriptions.append(formatted_transcription)

result = '\n\n'.join(formatted_transcriptions)
token_check = 0

response_schemas = [
    ResponseSchema(name="Video ID", description="Video ID of Youtube Link"),
    ResponseSchema(name="Overall Theme", description="Overall Theme of the Conversation"),
    ResponseSchema(name="Topics of Discussion", description="Different topics discussed in the transcription"),
    ResponseSchema(name="Category", description="High Level Category of Video based on content"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

chat_model = ChatOpenAI(model_name="gpt-4", temperature=0.5)

prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template("You are a helpful assistant who evaluates transcriptions of youtube videos, identifies topic of discussion and summarizes it in a concise format.\n{format_instructions}\nTranscriptions: {question}")
    ],
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)

topic_dict = {}
token_check = 0

for transcription in result.split('\n\n'):
  if num_tokens_from_string(transcription, "gpt-3.5-turbo")<8000:
    _input = prompt.format_prompt(question=transcription)
    output = chat_model(_input.to_messages())
    topic_dict[output_parser.parse(output.content)['Video ID']] = output_parser.parse(output.content)
    print(output_parser.parse(output.content))
  else:
    print("Token Limit Exceeded. Summarizing and evaluating")
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    summ_chain = load_summarize_chain(llm, chain_type="stuff")
    transcription = summ_chain.run(transcription)
    input = prompt.format_prompt(question=transcription)
    output = chat_model(_input.to_messages())
    topic_dict[output_parser.parse(output.content)['Video ID']] = output_parser.parse(output.content)
    print(output_parser.parse(output.content))

sentiment_output_custom_functions = [
    {
        'name': 'extract_sentiment_info',
        'description': 'Get topics driving positive and negative sentiments from the body of youtube comments',
        'parameters': {
            'type': 'object',
            'properties': {
                'positive_sentiment': {
                    'type': 'string',
                    'description': '{{topic 1: description, topic 2: description, ...}}'
                },
                'negative_sentiment': {
                    'type': 'string',
                    'description': '{{topic 1: description, topic 2: description, ...}}'
                }  
            }
        }
    }
]


for url in url_list[:30]:
  url = url
  comments = downloader.get_comments_from_url(url, sort_by=SORT_BY_POPULAR)
  comment_list = []

  urls_and_comments = {}

  for comment in islice(comments, 20):
    comment_list.append(comment['text'])

  urls_and_comments[url] = '\n'.join(comment_list)

  comments_chunk = str(list(urls_and_comments.values())[0])

  prompt2 = f'''
  Please extract the following information from the given youtube comments split by \n and return it as a JSON object:

  positive sentiment - {{topic 1: description, topic 2: description, ...}}
  negative sentiment - {{topic 1: description, topic 2: description, ...}}

  This is the body of text to extract the information from:
  {comments_chunk}
  '''

  response = openai.ChatCompletion.create(
      model = 'gpt-3.5-turbo',
      messages = [{'role': 'user', 'content': comments_chunk}],
      functions = sentiment_output_custom_functions,
      function_call = 'auto'
  )

  # Loading the response as a JSON object
  json_response = json.loads(response['choices'][0]['message']['function_call']['arguments'])
  print(json_response)

