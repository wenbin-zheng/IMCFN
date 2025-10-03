import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
from transformers import BertModel
from IMCFN.emotion_extractor.extract_emotion_ch import *
from IMCFN.emotion_extractor.extract_emotion_en import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class _LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(_LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class _SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(_SublayerConnection, self).__init__()
        self.norm = _LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_func):
        output_add = x + self.dropout(sublayer_func(x))
        return self.norm(output_add)

class _PositionwiseFeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(_PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class MyAttention(nn.Module):

    def __init__(self, feature_dim, attention_dim, ff_dim=2048, dropout=0.1):
        super(MyAttention, self).__init__()

        # --- 1. Attention W_Q, W_K, W_V 线性层 ---
        self.w_q = nn.Linear(feature_dim, attention_dim, bias=False)
        self.w_k = nn.Linear(feature_dim, attention_dim, bias=False)
        self.w_v = nn.Linear(feature_dim, attention_dim, bias=False)
        self.temperature = attention_dim ** 0.5

        self.feed_forward = _PositionwiseFeedForward(attention_dim, ff_dim, dropout)


        self.sublayer_attn = _SublayerConnection(attention_dim, dropout)
        self.sublayer_ffn = _SublayerConnection(attention_dim, dropout)

        self.feature_dim = feature_dim
        self.attention_dim = attention_dim

    def _compute_attention(self, q_feature, kv_feature):
        if q_feature.shape[-1] != self.attention_dim:
            pass

        q_3d = q_feature.unsqueeze(1)
        kv_3d = kv_feature.unsqueeze(1)

        q = self.w_q(q_3d)
        k = self.w_k(kv_3d)
        v = self.w_v(kv_3d)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.squeeze(1)

        return output

    def forward(self, q_feature, kv_feature=None):
            output = self.sublayer_attn(
                q_feature,
                lambda x: self._compute_attention(x, kv_feature)
            )

            return output


class CNN_Fusion(nn.Module):
    def __init__(self, args, shared_dim=128, sim_dim=64):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.event_num = args.event_num

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        # bert
        if self.args.dataset == 'weibo':
            bert_model = BertModel.from_pretrained('../chinese')
        else:
            bert_model = BertModel.from_pretrained('../uncased')

        self.bert_hidden_size = args.bert_hidden_dim
        self.shared_text_linear = nn.Sequential(
            nn.Linear(self.bert_hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )

        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        self.bertModel = bert_model

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        resnet = torchvision.models.resnet34(pretrained=True)

        num_ftrs = resnet.fc.out_features
        self.visualmodal = resnet
        self.shared_image = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU()
        )
        self.emotion_aligner = nn.Sequential(
            nn.Linear(in_features=47, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features=sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.attention_fusion = MyAttention(
            feature_dim=sim_dim,
            attention_dim=sim_dim,
            ff_dim=sim_dim * 4 
        )
         # fusion
        self.text_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )
        self.image_aligner = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.BatchNorm1d(shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU()
        )

        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)

        # Class  Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))

        self.unimodal_classifier = nn.Sequential(
            nn.Linear(sim_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax()
        )

        self.fusion = nn.Sequential(
            nn.Linear(sim_dim * 2, sim_dim * 2),
            nn.BatchNorm1d(sim_dim * 2),
            nn.ReLU(),
            nn.Linear(sim_dim * 2, sim_dim),
            nn.ReLU()
        )

        self.sim_classifier = nn.Sequential(
            nn.Linear(sim_dim * 3, sim_dim),
            nn.BatchNorm1d(sim_dim),
            nn.ReLU(),
            nn.Linear(sim_dim, 2)
        )

    def init_hidden(self, batch_size):
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x
    def get_emotion(self, ori_text):
        content = ori_text
        content_words = cut_words_from_text(content)
        sentiment = extract_publisher_emotion(content, content_words)
        sentiment = torch.Tensor(sentiment)

        return sentiment
    def forward(self, text, image, ela_image, ori_text, mask):
        image = self.visualmodal(image)
        image_z = self.shared_image(image)
        image_z = self.image_aligner(image_z)
        ela_image = self.visualmodal(ela_image)
        ela_z = self.shared_image(ela_image)
        ela_z = self.image_aligner(ela_z)

        image_z = self.attention_fusion(q_feature=ela_z, kv_feature=image_z)

        image_pred = self.unimodal_classifier(image_z)

        last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False,)

        text_z = self.shared_text_linear(last_hidden_state)
        text_z = self.text_aligner(text_z)

        sentiment_z = self.get_emotion(ori_text)
        sentiment_z = self.emotion_aligner(sentiment_z)

        text_z = self.attention_fusion(q_feature=sentiment_z, kv_feature=text_z)
        text_pred = self.unimodal_classifier(text_z)


        image_alpha = image_pred[:, -1] / (image_pred[:, -1] + text_pred[:, -1])  # normalize
        text_alpha = text_pred[:, -1] / (image_pred[:, -1] + text_pred[:, -1])

        image_alpha = image_alpha.unsqueeze(1)
        text_alpha = text_alpha.unsqueeze(1)

        text_image = torch.cat(((text_alpha * text_z), (image_alpha * image_z)), 1)
        text_image = self.fusion(text_image)

        text_image_fusion = torch.cat((text_z, text_image, image_z), 1)  # TODO: order?

        # Fake or real
        class_output = self.sim_classifier(text_image_fusion)

        class_output = self.dropout(class_output)
        return class_output, image_pred, text_pred, text_image, image_z, text_z

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)