import torch
import pandas as pd
import numpy as np
import os 
import time
from concurrent.futures import ThreadPoolExecutor

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from imagebind import data
from imagebind.data import load_and_transform_audio_data_tensors

from pinecone import Pinecone

from utils.datasets import StronglyLabelledDataset

device = "cuda:0" if torch.cuda.is_available() else "cpu"
PIPELINE_SIZE = 10000 # 10 thousand datapoints processed

# Instantiate ImageBind model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()

model.to(device)

# Connect to Pinecone index
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index("audioset-adorno-cv")

# Init dataset
dataset = StronglyLabelledDataset()

# Read ontology data for reference
onto = pd.read_csv('data/augmented_labels_cleaned.csv', index_col=0)
onto2 = pd.read_json('data/ontology.json')
onto = pd.merge(onto, onto2[['id','name']], on='name', how='left')

## Utils functions
def split_audio_in_segments(audio_tensor, audio_fps, labels_df):
    segments = []
    for i, row in labels_df.iterrows():
        segment = {}
        # Find indexes of start/end frames in the tensor
        start_frame = int(row['start_time_seconds'] * audio_fps)
        end_frame = int(row['end_time_seconds'] * audio_fps)
        segment_tensor = audio_tensor[:, start_frame:end_frame]
        length = row['end_time_seconds'] - row['start_time_seconds']

        segment['video_id'] = row['segment_id']
        segment['start_time'] = row['start_time_seconds']
        segment['end_time'] = row['end_time_seconds']
        segment['audio_tensor'] = segment_tensor
        segment['label_id'] = row['label']
        segment['type'] = row['MajorityType'] if pd.isnull(row['MajorityType'])==0 else 'Null'
                       
        segments.append(segment)
    return segments
    
def upload_data_with_metadata(data):
    for item_id, vector, meta in data:
        index.upsert(vectors=[(item_id, vector, meta)])

def process_for_pinecone(audio_embeddings, segments):
    i=0
    datapoints = []
    for embedding, audio_metadata in zip(audio_embeddings, segments):
        segment_id = audio_metadata['video_id'] + "_" + str(i)
        metadata = {key: audio_metadata[key] for key in ['start_time', 'end_time', 'label_id', 'type']}
        metadata['mode'] = 'audio'
        datapoint = (segment_id, embedding.cpu().tolist(), metadata) 
        datapoints.append(datapoint)
        i+=1

    return datapoints

def extract_data(index):
    video, audio, labels_df, info = dataset[index]
    return video, audio, labels_df, info

#### core logic
# Loop over dataset, embed, and upload to pinecone 
# Setup ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:  # AWS S3 has 10 max connections
    # Submit tasks to the executor
    futures = [executor.submit(extract_data, i) for i in range(PIPELINE_SIZE)]

    for future in futures:
        video, audio, labels_df, info = future.result()

        labels_df = pd.merge(labels_df, onto, how='left', left_on='label', right_on='id')
        labels_df = labels_df[labels_df['MajorityType'] != 'MUS']

        segments = split_audio_in_segments(audio, info['audio_fps'], labels_df)
        segment_tensors = [item['audio_tensor'] for item in segments]
        
        fps_list = [info['audio_fps'] for _ in range(len(segment_tensors))]
        try:
            transformed_segments = load_and_transform_audio_data_tensors(
                segment_tensors, fps_list, device='cuda'
            )
        except Exception as e:
            print(e)
            print(f'Segments: {segments}')

        inputs = {ModalityType.AUDIO: transformed_segments.to(device)}

        with torch.no_grad():
            outputs = model(inputs)
        audio_embeddings = outputs['audio']
        
        datapoints = process_for_pinecone(audio_embeddings, segments)
        
        upload_data_with_metadata(datapoints)
