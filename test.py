import torch
from BiLSTM_CRF import BiLSTM_CRF

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
ix_to_tag = {0 : "B", 1:  "I", 2: "O", 3: START_TAG, 4: STOP_TAG}

word_to_ix = {}
with open("word_to_ix.txt", "r", encoding="utf8") as reader:
    for line in reader:
        word, index, trash = line.split(" ")
        word_to_ix[word] = int(index)

#with open("model.pt", 'rb') as f:
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load("model.pt"))
model.eval()

test_data = ["georgia tech is in the wall street journal".split()]

with open("test.txt", "r", encoding="utf8") as reader:
    test_data = []
    sentence = []
    tags = []
    for line in reader:
        #print(line)
        if line!="\n":
            number, word, tag = line.replace("\n","").split("	")
            sentence.append(word)
            tags.append(tag)
        else:
            test_data.append((sentence,tags))
            sentence = []
            tags = []

with torch.no_grad():
    with open("test_graded.txt", "w", encoding="utf8") as writer:
        for (sentence, tags) in test_data:
            print(sentence)
            test_sent = prepare_sequence(sentence, word_to_ix)
            grades = model(test_sent)
            print(grades)
            sent_index = 0
            for word in sentence:
                tag = grades[1][sent_index]
                sent_index += 1
                writer.write(str(sent_index) + "	" + word + "	" + ix_to_tag[tag] +"\n")
            writer.write("\n")
