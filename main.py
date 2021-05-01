# Author: Robert Guthrie, from pytorch tutorial website
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
import argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from BiLSTM_CRF import BiLSTM_CRF

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--epoch', type=str, default='40',
                    help='number of epochs')
parser.add_argument('--seed', type=str, default='2',
                    help='random seed')
parser.add_argument('--train', type=str, default='train.txt',
                    help='path to train file')
parser.add_argument('--test', type=str, default='S21-gene-test.txt',
                    help='path to train file')
parser.add_argument('--save', type=str, default='model1.pt',
                    help='path to save file')
parser.add_argument('--hidden', type=str, default='4',
                    help='size of hidden layer')
parser.add_argument('--embedding', type=str, default='5',
                    help='size of embedding')
parser.add_argument('--load', type=str, default='',
                    help='load and continue training')
args = parser.parse_args()

torch.manual_seed(args.seed)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = int(args.embedding)
HIDDEN_DIM = int(args.hidden)

# loading training data and tags
with open(args.train, "r", encoding="utf8") as reader:
    training_data = []
    sentence = []
    tags = []
    for line in reader:
        if line != "\n":
            number, word, tag = line.replace("\n", "").split("	")
            sentence.append(word)
            tags.append(tag)
        else:
            training_data.append((sentence, tags))
            sentence = []
            tags = []

with open(args.test, "r", encoding="utf8") as reader:
    test_data = []
    sentence = []
    tags = []
    for line in reader:
        if line != "\n":
            number, word = line.replace("\n", "").split("	")
            sentence.append(word)
            tags.append(tag)
        else:
            test_data.append((sentence))
            sentence = []
            tags = []

# creating vocab dictionary
word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

for sentence in test_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

with open("word_to_ix.txt", "w", encoding="utf8") as writer:
    for item in word_to_ix:
        writer.write(item + " " + str(word_to_ix[item]) + " \n")

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

# creating model, as declared in BiLSTM_CRF.py
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM)
if args.load != "":
    print("loading")
    model.load_state_dict(torch.load(args.load))
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
# with torch.no_grad():
#     precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
#     precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
#     print(model(precheck_sent))

print("training")
for epoch in range(int(args.epoch)):
    for sentence, tags in training_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()
    print("Finished Epoch " + str(epoch))
    torch.save(model.state_dict(), args.save)

torch.save(model.state_dict(), args.save)

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
