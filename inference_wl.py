
from transformers import Wav2Vec2FeatureExtractor
import torch, librosa, gc
import json 
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder
import pandas as pd
from tqdm import tqdm
from dataloader import text_to_tensor
from MDD_model import Wav2Vec2_Error, Wav2Vec2_Linguistic
from pyctcdecode import build_ctcdecoder

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100

gc.collect()

df_dev = pd.read_csv("./test.csv")

model = Wav2Vec2_Linguistic.from_pretrained(
    'facebook/wav2vec2-base-100h', 
)
ckp = torch.load("checkpoint/checkpoint_wl.pth")
model.load_state_dict(ckp)
model.freeze_feature_extractor()
model = model.to(device)
PATH = []
CANONICAL = []
TRANSCRIPT = []
PREDICT = []
list_vocab = ['t ', 'n* ', 'y* ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ey* ', 'd* ', 'jh* ', 'ow ', 'aw ', 'ao* ', 'aa ', 'z* ', 'dh* ', 'aa* ', 'uw* ', 'th ', 'er* ', 'ih ', 't* ', 'zh ', 'g* ', 'k ', 'y ', 'l ', 'uh ', 'eh* ', 'p* ', 'ow* ', 'ch ', 'w ', 'b ', 'l* ', 'v ', 'ao ', 'w* ', 'aw* ', 'ah* ', 'uh* ', 'zh* ', 's ', 'k* ', 'p ', 'iy ', 'r ', 'ae* ', 'eh ', 'b* ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'hh* ', 'dh ', 'ae ', 'v* ', 'r* ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )
with torch.no_grad():
  model.eval().to(device)
  worderrorrate = []
  for point in tqdm(range(len(df_dev))):
    acoustic, _ = librosa.load("../EN_MDD/WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
    acoustic = feature_extractor(acoustic, sampling_rate = 16000)
    acoustic = torch.tensor(acoustic.input_values, device=device)
    transcript = df_dev['Transcript'][point]
    canonical = df_dev['Canonical'][point]
    canonical = text_to_tensor(canonical)
    canonical = torch.tensor(canonical, dtype=torch.long, device=device)
    logits = model(acoustic, canonical.unsqueeze(0))
    logits = F.log_softmax(logits.squeeze(0), dim=1)
    
    x = logits.detach().cpu().numpy()
    hypothesis = decoder_ctc.decode(x).strip()
    print(hypothesis)
    PATH.append(df_dev['Path'][point])
    CANONICAL.append(df_dev['Canonical'][point])
    TRANSCRIPT.append(df_dev['Transcript'][point])
    PREDICT.append(hypothesis)

train = pd.DataFrame([PATH, CANONICAL, TRANSCRIPT, PREDICT]) #Each list would be added as a row
train = train.transpose() #To Transpose and make each rows as columns
train.columns=['Path','Canonical', 'Transcript', 'Predict'] #Rename the columns
train.to_csv("result/W2v_l.csv")