import argparse
import torch
from BiLSTM_CRF import BiLSTM_CRF

parser = argparse.ArgumentParser(description='PyTorch NER LSTM Language Model')
parser.add_argument('--epoch', type=str, default='40',
                    help='number of epochs')
parser.add_argument('--seed', type=str, default='2',
                    help='random seed')
parser.add_argument('--test', type=str, default='S21-gene-test.txt',
                    help='path to test file')
parser.add_argument('--model', type=str, default='model-full.pt',
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
    unks = {}
    for word in seq:
        if word in to_ix:
            idxs.append(to_ix[word])
        else:
            idxs.append(to_ix["unk"])
    return torch.tensor(idxs, dtype=torch.long)


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = int(args.embedding)
HIDDEN_DIM = int(args.hidden)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
ix_to_tag = {0 : "B", 1:  "I", 2: "O", 3: START_TAG, 4: STOP_TAG}

word_to_ix = {}
# with open("word_to_ix.txt", "r", encoding="utf8") as reader:
#     for line in reader:
#         word, index, trash = line.replace("\n", "").split(" ")
#         word_to_ix[word] = int(index)
#     word_to_ix["unk"] = 31328

with open("S21-gene-train.txt", "r", encoding="utf8") as reader:
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

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(args.model))
model.eval()

with open(args.test, "r", encoding="utf8") as reader:
    test_data = []
    sentence = []
    for line in reader:
        if line != "\n":
            number, word = line.replace("\n", "").split("	")
            sentence.append(word)
        else:
            test_data.append((sentence))
            sentence = []

with torch.no_grad():
    with open(args.save, "w", encoding="utf8") as writer:
        for sentence in test_data:
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
