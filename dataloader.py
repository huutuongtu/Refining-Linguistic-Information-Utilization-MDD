import torch
from torch.utils.data import Dataset
import numpy as np
import json
import librosa
import ast
from transformers import Wav2Vec2FeatureExtractor
import json

with open('vocab.json') as f:
    dict_vocab = json.load(f)
key_list = list(dict_vocab.keys())
val_list = list(dict_vocab.values())

def text_to_tensor(string_text):
    text = string_text
    text = text.split(" ")
    text_list = []
    for idex in text:
        text_list.append(dict_vocab[idex])
    return text_list


# For error
class MDD_Dataset(Dataset):

    def __init__(self, data):
        self.len_data           = len(data)
        self.path               = list(data['Path'])
        self.canonical          = list(data['Canonical'])
        self.transcript         = list(data['Transcript'])
        self.error = list(data['Error'])
        self.canonical_time = list(data['Canonical_time'])

    def __getitem__(self, index):
        waveform, _ = librosa.load("../EN_MDD/WAV/" + self.path[index] + ".wav", sr=16000)
        linguistic  = text_to_tensor(self.canonical[index])
        transcript  = text_to_tensor(self.transcript[index])
        error = self.error[index]
        canonical_time = self.canonical_time[index]
        return waveform, linguistic, transcript, error, canonical_time

    def __len__(self):
        return self.len_data

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def collate_fn(batch):
    
    with torch.no_grad():    
        sr = 16000
        max_col = [-1] * 4
        target_length = []
        for row in batch:
            error = ast.literal_eval(row[3])
            # error = json.loads(error)
            if row[0].shape[0] > max_col[0]:
                max_col[0] = row[0].shape[0]
            if len(row[1]) > max_col[1]:
                max_col[1] = len(row[1])
            if len(row[2]) > max_col[2]:
                max_col[2] = len(row[2])
            if len(error) > max_col[3]:
                max_col[3] = len(error)

        cols = {'waveform':[], 'linguistic':[], 'transcript':[], 'error':[], 'outputlengths':[]}
        
        for row in batch:
            pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
            cols['waveform'].append(pad_wav)
            row[1].extend([68] * (max_col[1] - len(row[1])))
            cols['linguistic'].append(row[1])
            cols['outputlengths'].append(len(row[2]))
            row[2].extend([68] * (max_col[2] - len(row[2])))
            cols['transcript'].append(row[2])
            error = ast.literal_eval(row[3])
            error.extend([2] * (max_col[3] - len(error)))
            cols['error'].append(error)
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['linguistic'] = torch.tensor(cols['linguistic'], dtype=torch.long, device=device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
        cols['error'] = torch.tensor(cols['error'], dtype=torch.long, device=device)
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['linguistic'], cols['transcript'], cols['error'], cols['outputlengths']
