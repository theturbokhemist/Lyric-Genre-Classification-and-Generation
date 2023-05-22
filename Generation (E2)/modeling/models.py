import torch

class RNN(torch.nn.Module):
    def __init__(
            self, 
            embedding_dim, hidden_dim, vocab_size, num_layers = 2,
            pretrained_embeddings = None, # will initialize with random embeddings if None
            dropout = 0.1,
        ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim = embedding_dim)
        if(pretrained_embeddings != None):
            self.word_embeddings.weight = torch.nn.Parameter(pretrained_embeddings)
        self.dropout = torch.nn.Dropout(dropout)
        self.rnn = torch.nn.RNN(
            input_size = embedding_dim, 
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
        )
        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input, hidden):
        embeds = self.word_embeddings(input)
        embeds = self.dropout(embeds)
        output, hidden = self.rnn(embeds, hidden)
        decoded = self.decoder(output)
        return decoded, hidden
    

class LSTM(torch.nn.Module):
    def __init__(
            self, 
            embedding_dim, hidden_dim, vocab_size, num_layers = 2,
            pretrained_embeddings = None, # will initialize with random embeddings if None
            dropout = 0.1,
        ):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim = embedding_dim)
        if(pretrained_embeddings != None):
            self.word_embeddings.weight = torch.nn.Parameter(pretrained_embeddings)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size = embedding_dim, 
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
        )
        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input, hidden):
        embeds = self.word_embeddings(input)
        embeds = self.dropout(embeds)
        output, hidden = self.lstm(embeds, hidden)
        decoded = self.decoder(output)
        return decoded, hidden
    

class GRU(torch.nn.Module):
    def __init__(
            self, 
            embedding_dim, hidden_dim, vocab_size, num_layers = 2,
            pretrained_embeddings = None, # will initialize with random embeddings if None
            dropout = 0.1,
        ):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim = embedding_dim)
        if(pretrained_embeddings != None):
            self.word_embeddings.weight = torch.nn.Parameter(pretrained_embeddings)
        self.dropout = torch.nn.Dropout(dropout)
        self.gru = torch.nn.GRU(
            input_size = embedding_dim, 
            hidden_size = hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
        )
        self.decoder = torch.nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, input, hidden):
        embeds = self.word_embeddings(input)
        embeds = self.dropout(embeds)
        output, hidden = self.gru(embeds, hidden)
        decoded = self.decoder(output)
        return decoded, hidden