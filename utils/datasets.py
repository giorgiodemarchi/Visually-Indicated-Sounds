import pandas as pd
import boto3
from io import BytesIO
from torch.utils.data import Dataset
from torchvision.io import read_video
import tempfile
import os
from torchvision.transforms import Compose, Lambda
import torch
import torchvision.transforms.functional as F

class StronglyLabelledDataset(Dataset):
    """
    Video-audio dataset
    """
    def __init__(self, set='train', transform=None):
        """
        Custom dataset initializer.
        :param set: The dataset split, e.g. 'train', 'test', 'val'.
        :param transform: Transformations to be applied to the video frames.
        """
        self.transform = Compose([
            Lambda(lambda video: [F.to_pil_image(frame) for frame in video]),  # Convert each frame to PIL Image
            Lambda(lambda frames: [F.resize(frame, size=(224, 224)) for frame in frames]),  # Resize each frame
            Lambda(lambda frames: torch.stack([F.to_tensor(frame) for frame in frames])),  # Convert frames to tensors and stack
            Lambda(lambda frames: torch.stack([F.normalize(frame, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) for frame in frames]))  # Normalize and stack
        ])
        
        # Read credentials        
        AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
        AWS_ACCESS_KEY = os.environ.get('AWS_ACCESS_KEY')
        self.bucket_name = os.environ.get('BUCKET_NAME')

        self.s3_client = boto3.client('s3',
                                      aws_access_key_id=AWS_ACCESS_KEY,
                                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        
        # List of folders (datapoints)
        self.directory_name = set + "_strong/"
        self.folders = self.get_folder_names(set + '_strong/')
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.folders)
    
    def __getitem__(self, idx):
        """
        Loads and returns a sample from the dataset at the given index `idx`.
        """
        folder = self.folders[idx]
        id = folder.strip('_')[0]
        
        # Keys for video and metadata
        video_key = f"{self.directory_name}{folder}/video.mp4"
        metadata_key = f"{self.directory_name}{folder}/metadata.csv"

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:      
            # Load video into temporary file
            video_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=video_key)
            tmpfile.write(video_object['Body'].read())
            tmpfile.flush()
            
            # Use the name of the temporary file to load video and audio with read_video
            video, audio, info = read_video(tmpfile.name)
        
        # Now, the temporary file can be deleted as video and audio are already loaded
        os.unlink(tmpfile.name)

        video = video.permute(0, 3, 1, 2)
        if self.transform:
            video = self.transform(video)

        metadata_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=metadata_key)
        metadata_content = BytesIO(metadata_object['Body'].read())
        metadata_df = pd.read_csv(metadata_content)

        return video, audio, metadata_df
    
    def get_folder_names(self, directory_name):
        """
        Read all folder (datapoints) names in the images dataset
        """

        paginator = self.s3_client.get_paginator('list_objects_v2')
        response_iterator = paginator.paginate(
            Bucket=self.bucket_name,
            Prefix=directory_name,
            Delimiter='/'
        )

        folder_names = []
        for response in response_iterator:
            if response.get('CommonPrefixes') is not None:
                for prefix in response.get('CommonPrefixes'):
                    # Extract the folder name from the prefix key
                    folder_name = prefix.get('Prefix')
                    # Removing the base directory and the trailing slash to get the folder name
                    folder_name = folder_name[len(directory_name):].strip('/')
                    folder_names.append(folder_name)

        return folder_names




