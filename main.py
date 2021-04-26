# Author: Robert Guthrie, from pytorch tutorial website
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from BiLSTM_CRF import BiLSTM_CRF

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]


test_data = ["georgia tech is in the wall street journal".split()]

with open("train.txt", "r", encoding="utf8") as reader:
    training_data = []
    sentence = []
    tags = []
    for line in reader:
        #print(line)
        if line!="\n":
            number, word, tag = line.replace("\n","").split("	")
            sentence.append(word)
            tags.append(tag)
        else:
            training_data.append((sentence,tags))
            sentence = []
            tags = []


word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, START_TAG, STOP_TAG, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

#Make sure prepare_sequence from earlier in the LSTM section is loaded
print("training")
for epoch in range(25):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    print("Finished Epoch " + str(epoch))

torch.save(model.state_dict(), "model.pt")
with open("word_to_ix.txt", "w", encoding="utf8") as writer:
    for item in word_to_ix:
        writer.write(item + " " + str(word_to_ix[item]) + " \n")
# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!