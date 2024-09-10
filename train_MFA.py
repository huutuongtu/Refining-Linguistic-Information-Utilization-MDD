
from jiwer import wer
from transformers import Wav2Vec2FeatureExtractor
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
import numpy as np
from tqdm import tqdm
from dataloader import MDD_Dataset
from MDD_model import MFA_Wav2Vec2_Linguistic
from pyctcdecode import build_ctcdecoder
import ast

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100
batch_size = 4

gc.collect()
with open('vocab.json') as adsfs:
    dict_vocab = json.load(adsfs)


def collate_fn(batch):
    
    with torch.no_grad():
        
        sr = 16000
        max_col = [-1] * 4
        target_length = []
        for row in batch:
            error = ast.literal_eval(row[3])
            if row[0].shape[0] > max_col[0]:
                max_col[0] = row[0].shape[0]
            if len(row[1]) > max_col[1]:
                max_col[1] = len(row[1])
            if len(row[2]) > max_col[2]:
                max_col[2] = len(row[2])
            if len(error) > max_col[3]:
                max_col[3] = len(error)

        cols = {'waveform':[], 'linguistic':[], 'transcript':[], 'error':[], 'outputlengths':[], 'canonical_time':[]}
        
        max_length = max_col[0]//320
        for row in batch:
            canonical_time = torch.full((max_length,), 68)
            row_4 = ast.literal_eval(row[4])
            for i in range(len(row_4)):
                [(start, end)] = list(row_4[i].keys())
                [id] = list(row_4[i].values())
                id = dict_vocab[id]
                canonical_time[start:end] = id
            cols['canonical_time'].append(canonical_time)
            pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
            cols['waveform'].append(pad_wav)
            row[1].extend([68] * (max_col[1] - len(row[1])))
            cols['linguistic'].append(row[1])
            cols['outputlengths'].append(len(row[2]))
            row[2].extend([68] * (max_col[2] - len(row[2])))
            cols['transcript'].append(row[2])
            error.extend([2] * (max_col[3] - len(error)))
            cols['error'].append(error)
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['linguistic'] = torch.tensor(cols['linguistic'], dtype=torch.long, device=device)
        cols['canonical_time'] = torch.stack(cols['canonical_time']).to(device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
        cols['error'] = torch.tensor(cols['error'], dtype=torch.long, device=device)
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['linguistic'], cols['transcript'], cols['error'], cols['outputlengths'], cols['canonical_time']
  
df_train = pd.read_csv('./train_time.csv')
df_dev = pd.read_csv("./dev_time.csv")
train_dataset = MDD_Dataset(df_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
model = MFA_Wav2Vec2_Linguistic.from_pretrained(
    'facebook/wav2vec2-base-100h', 
)
model.freeze_feature_extractor()
model = model.to(device)

list_vocab = ['t ', 'n* ', 'y* ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ey* ', 'd* ', 'jh* ', 'ow ', 'aw ', 'ao* ', 'aa ', 'z* ', 'dh* ', 'aa* ', 'uw* ', 'th ', 'er* ', 'ih ', 't* ', 'zh ', 'g* ', 'k ', 'y ', 'l ', 'uh ', 'eh* ', 'p* ', 'ow* ', 'ch ', 'w ', 'b ', 'l* ', 'v ', 'ao ', 'w* ', 'aw* ', 'ah* ', 'uh* ', 'zh* ', 's ', 'k* ', 'p ', 'iy ', 'r ', 'ae* ', 'eh ', 'b* ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'hh* ', 'dh ', 'ae ', 'v* ', 'r* ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
ctc_loss = nn.CTCLoss(blank = 68)

for epoch in range(num_epoch):
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, _, labels, error_gt, target_lengths, linguistic  = data
    output = labels
    transcript = labels
    logits= model(acoustic, linguistic)
    logits = logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    loss_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    loss = loss_ctc
    running_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # break
  # scheduler.step()
  print(f"Training loss: {sum(running_loss) / len(running_loss)}")
  if epoch>=5:
    with torch.no_grad():
      model.eval().to(device)
      worderrorrate = []
      for point in tqdm(range(len(df_dev))):
        acoustic, _ = librosa.load("../EN_MDD/WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]

        canonical = df_dev['Canonical_time'][point]
        canonical_time = torch.full((acoustic.shape[1]//320,), 68).to(device)
        canonical = ast.literal_eval(canonical)
        for i in range(len(canonical)):

            [(start, end)] = list(canonical[i].keys())
            [id] = list(canonical[i].values())
            id = dict_vocab[id]
            canonical_time[start:end] = id

        logits = model(acoustic, canonical_time.unsqueeze(0))
        logits = F.log_softmax(logits.squeeze(0), dim=1)
        x = logits.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)
      epoch_wer = sum(worderrorrate)/len(worderrorrate)
      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        torch.save(model.state_dict(), 'checkpoint/checkpoint_MFA.pth')
      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
