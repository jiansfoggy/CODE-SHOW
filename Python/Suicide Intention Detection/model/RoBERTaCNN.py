from torch import nn
import numpy as np
# baseline part

class RoBERTaCNN(nn.Module):
    def __init__(self, roberta, num_classes, cnn_out_channels=100, cnn_kernel_size=2, cnn_stride=1):
        super(RoBERTaCNN, self).__init__()
        self.roberta = roberta
        self.embedding_dim = roberta.config.hidden_size
        self.cnn = nn.Conv1d(self.embedding_dim, cnn_out_channels, kernel_size=cnn_kernel_size, stride=cnn_stride)
        self.fc = nn.Linear(cnn_out_channels, num_classes)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from RoBERTa # [85, 64]
        embeddings = self.roberta(input_ids, attention_mask).last_hidden_state # [85, 64, 768]
        embeddings = embeddings.permute(0, 2, 1) # swap dimensions for CNN   # [85, 768, 64]

        # Apply CNN   
        cnn_features = self.cnn(embeddings)
        cnn_features = nn.functional.relu(cnn_features) # [85, 100, 63]
        cnn_features = nn.functional.max_pool1d(cnn_features, kernel_size=cnn_features.shape[-1]) # [85, 100, 1]
        cnn_features = cnn_features.squeeze(dim=-1)  # [85, 100]
        # Apply fully connected layer
        output = self.fc(cnn_features) # [85, 2]
        return output