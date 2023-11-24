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

class YTstats:

    def __init__(self, api_key, channel_id):
        self.api_key = api_key
        self.channel_id = channel_id
        self.channel_statistics = None
        self.video_data = None

    def extract_all(self):
        self.get_channel_statistics()
        self.get_channel_video_data()

    def get_channel_statistics(self):
        """Extract the channel statistics"""
        print('get channel statistics...')
        url = f'https://www.googleapis.com/youtube/v3/channels?part=statistics&id={self.channel_id}&key={self.api_key}'
        pbar = tqdm(total=1)

        json_url = requests.get(url)
        data = json.loads(json_url.text)
        try:
            data = data['items'][0]['statistics']
        except KeyError:
            print('Could not get channel statistics')
            data = {}

        self.channel_statistics = data
        pbar.update()
        pbar.close()
        return data

    def get_channel_video_data(self):
        "Extract all video information of the channel"
        print('get video data...')
        channel_videos, channel_playlists = self._get_channel_content(limit=50)

        parts=["snippet", "statistics","contentDetails", "topicDetails"]
        for video_id in tqdm(channel_videos):
            for part in parts:
                data = self._get_single_video_data(video_id, part)
                channel_videos[video_id].update(data)

        self.video_data = channel_videos
        return channel_videos

    def _get_single_video_data(self, video_id, part):
        """
        Extract further information for a single video
        parts can be: 'snippet', 'statistics', 'contentDetails', 'topicDetails'
        """

        url = f"https://www.googleapis.com/youtube/v3/videos?part={part}&id={video_id}&key={self.api_key}"
        json_url = requests.get(url)
        data = json.loads(json_url.text)
        try:
            data = data['items'][0][part]
        except KeyError as e:
            print(f'Error! Could not get {part} part of data: \n{data}')
            data = dict()
        return data

    def _get_channel_content(self, limit=None, check_all_pages=True):
        """
        Extract all videos and playlists, can check all available search pages
        channel_videos = videoId: title, publishedAt
        channel_playlists = playlistId: title, publishedAt
        return channel_videos, channel_playlists
        """
        url = f"https://www.googleapis.com/youtube/v3/search?key={self.api_key}&channelId={self.channel_id}&part=snippet,id&order=date"
        if limit is not None and isinstance(limit, int):
            url += "&maxResults=" + str(limit)

        vid, pl, npt = self._get_channel_content_per_page(url)
        idx = 0
        while(check_all_pages and npt is not None and idx < 10):
            nexturl = url + "&pageToken=" + npt
            next_vid, next_pl, npt = self._get_channel_content_per_page(nexturl)
            vid.update(next_vid)
            pl.update(next_pl)
            idx += 1

        return vid, pl

    def _get_channel_content_per_page(self, url):
        """
        Extract all videos and playlists per page
        return channel_videos, channel_playlists, nextPageToken
        """
        json_url = requests.get(url)
        data = json.loads(json_url.text)
        channel_videos = dict()
        channel_playlists = dict()
        if 'items' not in data:
            print('Error! Could not get correct channel data!\n', data)
            return channel_videos, channel_videos, None

        nextPageToken = data.get("nextPageToken", None)

        item_data = data['items']
        for item in item_data:
            try:
                kind = item['id']['kind']
                published_at = item['snippet']['publishedAt']
                title = item['snippet']['title']
                if kind == 'youtube#video':
                    video_id = item['id']['videoId']
                    channel_videos[video_id] = {'publishedAt': published_at, 'title': title}
                elif kind == 'youtube#playlist':
                    playlist_id = item['id']['playlistId']
                    channel_playlists[playlist_id] = {'publishedAt': published_at, 'title': title}
            except KeyError as e:
                print('Error! Could not extract data from item:\n', item)

        return channel_videos, channel_playlists, nextPageToken

    def dump(self):
        """Dumps channel statistics and video data in a single json file"""
        if self.channel_statistics is None or self.video_data is None:
            print('data is missing!\nCall get_channel_statistics() and get_channel_video_data() first!')
            return

        fused_data = {self.channel_id: {"channel_statistics": self.channel_statistics,
                              "video_data": self.video_data}}

        channel_title = self.video_data.popitem()[1].get('channelTitle', self.channel_id)
        channel_title = channel_title.replace(" ", "_").lower()
        filename = channel_title + '.json'
        with open('/content/'+str(filename), 'w') as f:
            json.dump(fused_data, f, indent=4)

        print('file dumped to', filename)

    def to_dataframe(self):
        """
        Convert video data to a pandas DataFrame.
        """
        if self.video_data is None:
            print('Video data is missing! Call get_channel_video_data() first!')
            return None

        # Flatten the data and convert it to a list of dictionaries
        video_items = []
        for video_id, data in self.video_data.items():
            data['video_id'] = video_id
            video_items.append(data)

        # Create DataFrame
        df = pd.DataFrame(video_items)
        return df

def split_into_chunks(text, chunk_size=8000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def url_extractor_by_channel(channel_id):
  url_list = []
  #Extract URL's for a given channel
  videos = scrapetube.get_channel(channel_id)
  for video in videos:
      url_list.append("https://www.youtube.com/watch?v="+str(video['videoId']))
  return url_list

def video_comments(api_key,video_id):
    # List to store comments
    comments_list = []

    # Creating YouTube resource object
    youtube = build('youtube', 'v3', developerKey=api_key)

    # Retrieve YouTube video results
    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id
    ).execute()

    # Iterate video response
    while video_response:
        for item in video_response['items']:
            # Extracting comment
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments_list.append(comment)

            # Check if there are replies to the comment
            replycount = item['snippet']['totalReplyCount']
            if replycount > 0:
                # Iterate through all replies
                for reply in item['replies']['comments']:
                    # Extract and add reply to the list
                    reply_text = reply['snippet']['textDisplay']
                    comments_list.append(reply_text)

        # Check if there are more pages
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                pageToken=video_response['nextPageToken']
            ).execute()
        else:
            break

    return comments_list

def url_transcriber(url_list, output_path):
  #for i in range(len(url_list)):
  url_mp3_mapping = {}
  for i in range(len(url_list)):
    yt = YouTube(url_list[i])
    video = yt.streams.filter(only_audio=True).first()
    destination = output_path
    out_file = video.download(output_path=destination)
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    url_mp3_mapping[new_file] = url_list[i]
    os.rename(out_file, new_file)
    print(yt.title + " has been successfully downloaded.")
  return url_mp3_mapping

def transcriber(audio_file_path, model):
  segments, info = model.transcribe(audio_file_path, beam_size=5)

  print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

  transcription = []

  for segment in segments:
      print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
      transcription.append(segment.text)
  return(transcription)

def topic_identification_comments(comments_chunk):
  # Define the system message
  system_msg = 'You are a helpful assistant who evaluates comments on youtube videos, identifies topic of discussion, sentiment and summarizes it in a concise format.'

  # Define the user message
  user_msg = """
  Extract high level topics of conversation from the following comments:

  **
  {}
  **

  Return the output as bullet points in the following format:
  - topic 1: description, sentiment
  - topic 2: description, sentiment
  - topic 3: description, sentiment

  Do not make up the answer if you do not know.

  """

  final_df = pd.DataFrame(columns=['Topic','Sentiment'])

  for chunk in chunks:
    reqd_user_msg = user_msg.format(comments_chunk)
    response = openai.ChatCompletion.create(model="gpt-4",
                                            temperature=0.5,
                                            max_tokens=500,
                                            top_p=1,
                                            frequency_penalty=0.5,
                                            presence_penalty=0,
                                            messages=[{"role": "system", "content": system_msg},
                                            {"role": "user", "content": reqd_user_msg}])

    flattened_list = " ".join(response.choices[0].message.content.split("\n  \n"))
    topics = re.split(r'- Topic \d+: ', flattened_list)[1:]  # Split and remove the first empty string

    data = []

    for topic in topics:
      title_search = re.search(r'^[\w\s]+:', topic)
      sentiment_search = re.search(r'Sentiment is .+?(?=\.- Topic|$)', topic)
      title = title_search.group()[:-1] if title_search else "Unknown Title"
      sentiment = sentiment_search.group() if sentiment_search else "Unknown Sentiment"
      data.append({'Topic': title, 'Sentiment': sentiment})

    comments_df = pd.DataFrame(data)
    final_df = pd.concat([final_df,comments_df])

  # Define the system message
  system_msg = 'You are a helpful assistant that removes redundant, combines and generates high level categories and sub categories from a given list of topics'

  # Define the user message
  user_msg = """
  Generate high level and sub categories of the following topics split by ',':

  ""
  {}
  ""

  Return the output as bullet points in the following format:

  ""
  - High Level Category 1: sub category1, sub category2, sub category3,...
  - High Level Category 2: sub category1, sub category2, sub category3,...
  - High Level Category 3: sub category1, sub category2, sub category3,...
  ...
  ""
  Do not make up the answer if you do not know.

  """
  response = openai.ChatCompletion.create(model="gpt-4",
                                          temperature=0.5,
                                          max_tokens=500,
                                          top_p=1,
                                          frequency_penalty=0.5,
                                          presence_penalty=0,
                                          messages=[{"role": "system", "content": system_msg},
                                          {"role": "user", "content": user_msg.format(', '.join(final_df['Topic']))}])

  return response.choices[0].message.content.split('\n')

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to extract minutes from duration strings
def extract_minutes(duration_str):
    match = re.match(r'PT(\d+)M', duration_str)
    if match:
        return int(match.group(1))
    else:
        return 0

def transcription_evaluator_output_parser(transcription_text):
  topic_df = pd.DataFrame(columns=['video_id','theme','topic_dict','phrases_common'])
  index = 0
  for i in range(0,len(transcription_text.replace("'",'').split('\n')),6):
    video_id = transcription_text.replace("'",'').split('\n')[i].split(' ')[1]
    theme = transcription_text.replace("'",'').split('\n')[i+1].split(': ')[1]
    topic_dict = transcription_text.replace("'",'').split('\n')[i+2].split(': {')[1].replace('}','').split(',')
    phrase_common = ast.literal_eval(transcription_text.replace("'",'').split('\n')[i+3].split(': ')[1])
    topic_df = topic_df.append({"video_id":video_id, "theme":theme, "topic_dict":topic_dict, "phrases_common":phrase_common}, ignore_index=True)
  return topic_df

def video_id_from_url(url):
  url_parsed = parse.urlparse(url)
  qsl = parse.parse_qs(url_parsed.query)
  return qsl['v'][0]

def predict_sentiment(text, tokenizer, model):
  # Load the pre-trained tokenizer and model
  tokenizer = AutoTokenizer.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
  model = AutoModelForSequenceClassification.from_pretrained("LiYuan/amazon-review-sentiment-analysis")
  sentiment_classes = ['Negative', 'Neutral', 'Positive']
  # Tokenize the input text
  inputs = tokenizer(text, return_tensors="pt")

  # Perform inference
  with torch.no_grad():
      outputs = model(**inputs)

  logits = outputs.logits
  predicted_sentiment = torch.argmax(logits, dim=1).item()
  # Ensure the predicted sentiment is within valid range
  if predicted_sentiment < 0 or predicted_sentiment >= len(sentiment_classes):
      return "Unknown"

  # Map the sentiment label to its corresponding class
  predicted_sentiment_label = sentiment_classes[predicted_sentiment]

  return predicted_sentiment_label

def topic_evaluation_df(result):
  system_msg = 'You are a helpful assistant who evaluates transcriptions of youtube videos, identifies theme, topic of discussion, commonly used phrases, number of speakers and summarizes it in a concise format.'

  # Define the user message
  user_msg = """
  Extract high level topics of conversation from the following transcriptions in the following format:

  ""
  Transcription <youtube_video_id_1>
  "transcription youtube_video_id_2"

  Transcription <youtube_video_id_2>
  "transcription youtube_video_id_2"

  ....
  ....
  ""

  **
  {}
  **

  Return the output as bullet points in the following format:

  ""
  -Transcription 1
    - Overall Theme1: overall theme and message of transcription1
    - dictionary of topics 1: {{topic1 : description1, topic2 : description2, ...}}
    - commonly used phrases 1: [list of commonly used phrases in transcription1]
    - number of speakers 1: estimated number of speakers in transcription1
    ...
  -Transcription 2
    - Overall Theme1: overall theme and message of transcription2
    - dictionary of topics 2: {{topic1 : description1, topic2 : description2, ...}}
    - commonly used phrases 2: [list of commonly used phrases in transcription2]
    - number of speakers 2: estimated number of speakers in transcription2
    ...
  ""

  Stick to the above format. Do not make up the answer if you do not know and be precise. Do not add any additional information in the output.

  """
  reqd_user_msg = user_msg.format(result)
  response = openai.ChatCompletion.create(model="gpt-4",
                                          temperature=0.5,
                                          max_tokens=2000,
                                          top_p=1,
                                          frequency_penalty=0.5,
                                          presence_penalty=0,
                                          messages=[{"role": "system", "content": system_msg},
                                          {"role": "user", "content": reqd_user_msg}])
  transcription_text = response.choices[0].message.content
  topic_df = transcription_evaluator_output_parser(transcription_text)
  return topic_df

def sentiment_output_parser(sentiment_output):
  sentiment_sections = re.findall(r'(\w+ sentiment): \{(.*?)\}', sentiment_output, re.DOTALL)
  sentiment_data = {}
  for sentiment, content in sentiment_sections:
      sentiment_data[sentiment] = content.split(', ')
  return(sentiment_data)

def sentiment_evaluation_df(comments_context):
  system_msg = 'You are a helpful assistant who evaluates comments on youtube videos, identifies topic of discussion, sentiment and summarizes it in a concise format.'

  # Define the user message
  user_msg = """
  Extract sentiment and high level aspect of sentiment as topics from the following comments on a youtube video:

  ""
  {}
  ""

  Return the output in the following format:
  ""
  positive sentiment: {{topic 1: description, topic 2: description, ...}}
  negative sentiment: {{topic 1: description, topic 2: description, ...}}
  ""

  Stick to the above format. Do not make up the answer if you do not know and be precise. Do not add any additional information in the output.

  """


  reqd_user_msg = user_msg.format(comments_context)
  response = openai.ChatCompletion.create(model="gpt-4",
                                          temperature=0.5,
                                          max_tokens=600,
                                          top_p=1,
                                          frequency_penalty=0.5,
                                          presence_penalty=0,
                                          messages=[{"role": "system", "content": system_msg},
                                          {"role": "user", "content": reqd_user_msg}])

  sentiment_output = response.choices[0].message.content
  sentiment_output = sentiment_output_parser(sentiment_output)
  return sentiment_output

# Open the text file for reading
file_path = '/content/sniper_wolf.txt'  # Replace 'urls.txt' with the path to your text file
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

