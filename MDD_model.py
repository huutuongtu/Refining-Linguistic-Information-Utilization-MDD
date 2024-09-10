import torch.nn as nn
import torch     
from transformers import Wav2Vec2PreTrainedModel, Wav2Vec2Model
import einops
          
class CNN_Stack(nn.Module):
    def __init__(self, num_features, p=0.2):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(num_features=num_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=p)

        #input BxTxC, return BxTxC
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv2d(x)  # -> batch_size x 1 x time x num_features
        x = x.squeeze(1)    # -> batch_size x time x num_features
        x = einops.rearrange(x, 'b t c -> b c t')
        x = self.bn(x)     
        x = self.relu(x)    
        x = self.dropout(x) 
        x = einops.rearrange(x, 'b c t -> b t c')
        return x
    
class RNN_Stack(nn.Module):
    def __init__(self, num_features_in, num_features_out, p=0.2):
        super().__init__()
        assert num_features_out % 2 == 0, 'num_features_out must be divided by 2'
        self.bi_lstm = nn.LSTM(
            input_size=num_features_in, hidden_size=num_features_out//2, bidirectional=True, batch_first=True
        )
        self.bn = nn.BatchNorm1d(num_features_out)
        self.dropout = nn.Dropout(p=p)
    #input batch x time x num_features, return BxTxC
    def forward(self, x):
        x, _    = self.bi_lstm(x)   # batch_size x time x num_features_out
        x = einops.rearrange(x, 'b t c -> b c t')
        x       = self.bn(x)
        x       = self.dropout(x)   
        x = einops.rearrange(x, 'b c t -> b t c')
        return x
    

class LinguisticEncoder(nn.Module):
    def __init__(self, num_features_out=768, vocab_size=68):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size+1, 64, padding_idx=vocab_size)
        self.bi_lstm    = nn.LSTM(
            input_size=64, hidden_size=num_features_out//2, bidirectional=True, 
            batch_first=True, num_layers=4
        )
        self.linear     = nn.Linear(num_features_out, num_features_out)

    def forward(self, x):
        # x shape : batch_size x length_phoneme output batch x length x 768
        x           = self.embedding(x)     # batch_size x length_phoneme x 64
        out, (h_n, c_n)   = self.bi_lstm(x)
        Hk          = self.linear(out)
        Hv          = out
        return Hk, Hv
      

#Linguistic - W2v
class Wav2Vec2_Linguistic(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier_vocab = nn.Linear(1536, 69)
        self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
        self.post_init()
        self.linguistic_encoder = LinguisticEncoder()
        
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input, canonical):
        acoustic = self.wav2vec2(audio_input, attention_mask=None)[0] #b x t x 768
        Hk, Hv = self.linguistic_encoder(canonical)
        o, _     = self.multihead_attention(acoustic, Hk, Hv)
        o        = torch.concat([acoustic, o], dim = 2)
        logits = self.classifier_vocab(o)
      
        return logits


class MFA_Wav2Vec2_Linguistic(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier_vocab = nn.Linear(1536, 69)
        self.multihead_attention_a = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
        self.multihead_attention_l = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
        self.prj_a = nn.Linear(768, 768)
        self.prj_l = nn.Linear(768, 768)
        self.embedding = nn.Embedding(69, 768, padding_idx=68)
        self.post_init()
        
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input, canonical):
        acoustic = self.wav2vec2(audio_input, attention_mask=None)[0] #b x t x 768
        linguistic = self.embedding(canonical)
        linguistic = linguistic[:, :acoustic.shape[1], :]
        a, _ = self.multihead_attention_a(acoustic, self.prj_l(linguistic), linguistic)
        l, _ = self.multihead_attention_l(linguistic, self.prj_a(acoustic), acoustic)
        o = torch.concat([a, l], dim = 2)
        logits = self.classifier_vocab(o)
        return logits  
    

#For error classifier classifier
class Wav2Vec2_Error(Wav2Vec2PreTrainedModel):
  def __init__(self, config):
      super().__init__(config)
      self.wav2vec2 = Wav2Vec2Model(config)
      self.classifier_vocab = nn.Linear(768, 69)
      self.linear1 = nn.Linear(768, 768)
      self.multihead_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
      self.compare_attention = nn.MultiheadAttention(embed_dim=768, num_heads=16, dropout=0.2, batch_first=True)
      self.post_init()
      self.embedding = nn.Embedding(69, 768, padding_idx=68)
      self.error_classifier = nn.Linear(768, 2)

  def freeze_feature_extractor(self):
      self.wav2vec2.feature_extractor._freeze_parameters()
      

  def forward(self, audio_input, canonical):
      acoustic = self.wav2vec2(audio_input, attention_mask=None)[0] #b x t x 768
      linguistic = self.embedding(canonical)
      linguistic_error, _ = self.compare_attention(linguistic, acoustic, acoustic)
      linguistic_error = linguistic - linguistic_error
      acoustic, _ = self.multihead_attention(acoustic, linguistic_error, linguistic_error)
      logits = self.classifier_vocab(acoustic)
      linguistic_error = self.error_classifier(linguistic_error)
      return logits, linguistic_error
        