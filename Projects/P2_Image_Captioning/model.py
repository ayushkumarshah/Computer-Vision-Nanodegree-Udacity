import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, 
                            num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax()
    
    def forward(self, features, captions):
        captions = captions[:, :-1] 
        captions = self.embed(captions)
        
        # Concatenate CNN output features form image and captions
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        out, _ = self.lstm(inputs)
        
        out = self.linear(out)
        return out

    def sample(self, inputs, hidden=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []   
        sentence_length = 0
        index = None
        while (sentence_length != max_len+1 and index != 1):
            
            output, hidden = self.lstm(inputs, hidden)
           
            output = self.linear(output.squeeze(dim = 1))
            _, index = torch.max(output, 1)
            
            outputs.append(index.cpu().numpy()[0].item())

            inputs = self.embed(index).unsqueeze(1)  
            
            sentence_length += 1

        return outputs