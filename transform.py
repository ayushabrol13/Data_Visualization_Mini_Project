import googleapiclient.discovery
import pandas as pd
import pycountry
from cleantext import clean
from langdetect import detect, LangDetectException
from textblob import TextBlob
import streamlit as st
import demoji
import unidecode


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


def get_sentiment(polarity):
    if polarity > 0:
        return 'POSITIVE'
    if polarity < 0:
        return 'NEGATIVE'
    return 'NEUTRAL'


def det_lang(language):
    try:
        lang = detect(language)

    except LangDetectException:
        lang = 'Other'
    return lang


def parse_video(url) -> pd.DataFrame:
    video_id = url.split('?v=')[-1]
    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=st.secrets["api_key"])

    video_response = youtube.commentThreads().list(
        part='snippet',
        maxResults=200,
        order='relevance',
        videoId=video_id
    ).execute()

    comments = []

    for requested_item in video_response['items']:

        comment = requested_item['snippet']['topLevelComment']['snippet']['textOriginal']
        author = requested_item['snippet']['topLevelComment']['snippet']['authorDisplayName']
        published_at = requested_item['snippet']['topLevelComment']['snippet']['publishedAt']
        like_count = requested_item['snippet']['topLevelComment']['snippet']['likeCount']
        reply_count = requested_item['snippet']['totalReplyCount']

        comments.append(
            [author, comment, published_at, like_count, reply_count])

    df_transform = pd.DataFrame({'Author': [i[0] for i in comments],
                                 'Comment': [i[1] for i in comments],
                                 'Timestamp': [i[2] for i in comments],
                                 'Likes': [i[3] for i in comments],
                                 'TotalReplies': [i[4] for i in comments]})

    df_transform['Comment'] = df_transform['Comment'].apply(
        lambda x: x.strip().lower().replace('xd', '').replace('<3', ''))

    df_transform['Comment'] = df_transform['Comment'].apply(
        lambda x: clean(x, clean_all=True, lowercase=True, punct=False))

    df_transform['Comment'] = df_transform['Comment'].apply(
        lambda x: demoji.replace(x, ''))

    df_transform['Comment'] = df_transform['Comment'].apply(
        lambda x: x.replace('"', '').replace("'", ''))

    df_transform['Comment'] = df_transform['Comment'].apply(
        lambda x: unidecode.unidecode(x))

    df_transform['Language'] = df_transform['Comment'].apply(det_lang)

    df_transform['Language'] = df_transform['Language'].apply(
        lambda x: pycountry.languages.get(alpha_2=x).name if (x) != 'Other' else 'Not-Detected')

    df_transform.drop(
        df_transform[df_transform['Language'] == 'Not-Detected'].index, inplace=True)

    df_transform['TextBlob_Polarity'] = df_transform[['Comment', 'Language']].apply(
        lambda x: get_polarity(x['Comment']) if x['Language'] == 'English' else '', axis=1)

    df_transform['TextBlob_Sentiment_Type'] = df_transform['TextBlob_Polarity'].apply(
        lambda x: get_sentiment(x) if isinstance(x, float) else '')

    df_transform['Timestamp'] = pd.to_datetime(
        df_transform['Timestamp']).dt.strftime('%Y-%m-%d %r')

    return df_transform


def youtube_metrics(url) -> list:
    video_id = url.split('?v=')[-1]

    youtube = googleapiclient.discovery.build(
        'youtube', 'v3', developerKey=st.secrets["api_key"])

    req_stats = youtube.videos().list(
        part="statistics",
        id=video_id
    ).execute()

    youtube_metrics_ = []

    for requested_item in req_stats['items']:

        youtube_metrics_.append(requested_item['statistics']['viewCount'])
        youtube_metrics_.append(requested_item['statistics']['likeCount'])
        youtube_metrics_.append(requested_item['statistics']['commentCount'])

    return youtube_metrics_


if __name__ == "__main__":
    dataframe_main = parse_video('https://www.youtube.com/watch?v=adLGHcj_fmA')
    dataframe_youtube = youtube_metrics(
        'https://www.youtube.com/watch?v=adLGHcj_fmA')
    print(dataframe_main.head())
    print(dataframe_youtube)
