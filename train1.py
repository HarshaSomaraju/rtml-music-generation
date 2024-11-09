from music21 import converter, instrument, note, chord
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm

import time

f=open('output_TF.txt','w')
f.close()

Path_to_midi = 'RTML Project Data'
notes = []
for file in glob.glob(Path_to_midi+'/*/*.mid'):
    midi = converter.parse(file)
    notes_to_parse = None
    
    parts = instrument.partitionByInstrument(midi)
    
    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
        
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

n_vocab=len(set(notes))


sequence_length = 100

# get all pitch names
pitchnames = sorted(set(item for item in notes))

# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

network_input = []
network_output = []

# create input sequences and the corresponding outputs

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
    
n_patterns = len(network_input)
    
# reshape the input into a format compatible with LSTM layers
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# normalize input
network_input = network_input / float(n_vocab)

class LSTMC(nn.Module):

    def __init__(self, hidden_size, sequence_length, n_vocab):
        super(LSTMC, self).__init__()
        self.hidden_size = hidden_size
        self.lstm=nn.LSTMCell(1,hidden_size).float()
        self.lstm2=nn.LSTMCell(hidden_size,hidden_size).float()
        self.lstm3=nn.LSTMCell(hidden_size,hidden_size).float()
        self.batch_normalization = nn.BatchNorm1d(num_features=hidden_size)
        self.batch_normalization2 = nn.BatchNorm1d(num_features=256)
        self.drop_out = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size,256)
        self.out_layer = nn.Linear(256,n_vocab)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        
    def init_hidden(self,sequence_length,hidden_size,batch_size):
        return (torch.randn(batch_size, hidden_size,device=device), torch.randn(batch_size, hidden_size,device=device),
                torch.randn(batch_size, hidden_size,device=device), torch.randn(batch_size, hidden_size,device=device),
                torch.randn(batch_size, hidden_size,device=device), torch.randn(batch_size, hidden_size,device=device))

    def forward(self, input_sequence,hidden):
        h1,c1 = self.lstm(input_sequence,(hidden[0],hidden[1]))
        h1 = self.drop_out(h1)
        h2,c2 = self.lstm2(h1,(hidden[2],hidden[3]))
        h2 = self.drop_out(h2)
        h3,c3 = self.lstm3(h2,(hidden[4],hidden[5]))
        if(h1.size(0)>1):
            output = self.batch_normalization(h3)
        else:
            output = h3
        output = self.drop_out(output)
        output = self.linear(output)
        output = self.relu(output)
        if(output.size(0)>1):
            output = self.batch_normalization2(output)
        output = self.drop_out(output)
        output = self.out_layer(output)
        output = self.softmax(output)
        return output

device = 'cuda:1'
n_i = torch.tensor(network_input,device=device).float()

lstm_cell_model = LSTMC(512,100,n_vocab)
lstm_cell_model.to(device)


criteron = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(lstm_cell_model.parameters(),lr=0.01)

def write_output(epoch,loss,time):
    with open('output_TF.txt','a') as f:
        f.write('Loss for epoch '+str(epoch)+' is '+str(loss)+'. Time taken is '+str(time)+' seconds.\n')

def train(batch_size,epochs):
    st = time.time()
    losses=[]
    lstm_cell_model.train()
    h_i = lstm_cell_model.init_hidden(100,512,batch_size=batch_size)
    for epoch in range(epochs):
        l=0
        i=0
        output = torch.rand(batch_size,n_vocab)
        while(i+batch_size<len(network_output)):
            lstm_cell_model.zero_grad()
            if(np.random.rand()<0.7 and i!=0):              #Teacher Forcing
                for k in range(sequence_length):
                    a=lstm_cell_model(input_sequence = n_i[i:i+batch_size,k,0:],hidden=h_i)
            else:
                for k in range(sequence_length-1):
                    a=lstm_cell_model(input_sequence = n_i[i:i+batch_size,k,0:],hidden=h_i)
                h_i = lstm_cell_model.init_hidden(100,512,batch_size=batch_size)
                a=lstm_cell_model(input_sequence = output.argmax(dim=1).unsqueeze(1).to(device).float(),hidden=h_i)
            output = a
            loss = criteron(a,torch.tensor(network_output[i:i+batch_size],device=device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            l+=loss.item()
            i=i+batch_size
        losses.append(l/(len(network_output)//batch_size))
        write_output(epoch,losses[-1],time.time()-st)
        if epoch==0 or epoch==1 or epoch==2:
            plt.plot(losses, 'b-')
            plt.xlabel('Epoch (%d)' % epoch)
            plt.ylabel('Loss')
            plt.savefig('Loss_TF_Epoch_'+str(epoch)+'.png')     
        if epoch%5==0:
            torch.save(lstm_cell_model.state_dict(),'lstm_cell_model_TF.pth')
        if epoch%25==0:
            plt.plot(losses, 'b-')
            plt.xlabel('Epoch (%d)' % epoch)
            plt.ylabel('Loss')
            plt.savefig('Loss_TF_Epoch_'+str(epoch)+'.png')

st = time.time()
train(32,200)
print("Time taken is: ",st-time.time())
       