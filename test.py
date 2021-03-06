import argparse
import torch
from BiLSTM_CRF import BiLSTM_CRF

parser = argparse.ArgumentParser(description='PyTorch NER LSTM Language Model')
parser.add_argument('--epoch', type=str, default='40',
                    help='number of epochs')
parser.add_argument('--seed', type=str, default='2',
                    help='random seed')
parser.add_argument('--test', type=str, default='train.txt',
                    help='path to test file')
parser.add_argument('--model', type=str, default='model1.pt',
                    help='path to saved model')
parser.add_argument('--save', type=str, default='test_graded.txt',
                    help='path to save graded file')
parser.add_argument('--hidden', type=str, default='4',
                    help='size of hidden layer')
parser.add_argument('--embedding', type=str, default='5',
                    help='size of embedding')
args = parser.parse_args()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = int(args.embedding)
HIDDEN_DIM = int(args.hidden)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
ix_to_tag = {0 : "B", 1:  "I", 2: "O", 3: START_TAG, 4: STOP_TAG}

word_to_ix = {}
with open("word_to_ix.txt", "r", encoding="utf8") as reader:
    for line in reader:
        word, index, trash = line.split(" ")
        word_to_ix[word] = int(index)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(args.model))
model.eval()

with open(args.test, "r", encoding="utf8") as reader:
    test_data = []
    sentence = []
    tags = []
    for line in reader:
        if line != "\n":
            number, word, tag = line.replace("\n", "").split("	")
            sentence.append(word)
            tags.append(tag)
        else:
            test_data.append((sentence,tags))
            sentence = []
            tags = []

with torch.no_grad():
    with open(args.save, "w", encoding="utf8") as writer:
        for (sentence, tags) in test_data:
            #print(sentence)
            test_sent = prepare_sequence(sentence, word_to_ix)
            grades = model(test_sent)
            #print(grades)
            sent_index = 0
            for word in sentence:
                tag = grades[1][sent_index]
                sent_index += 1
                writer.write(str(sent_index) + "	" + word + "	" + ix_to_tag[tag] + "\n")
            writer.write("\n")
