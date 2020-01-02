import json
import math
import time
import shutil
import torch
import torch.nn.functional as F
import torch.optim
import torch.optim as optim
import torch.utils.data as data
from nltk.tokenize.treebank import TreebankWordDetokenizer
from torchtext import data as torchtext_data
from torchtext import datasets
from tqdm import tqdm, trange

from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pplm_classification_head import ClassificationHead

import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt



model_size = "medium"
model_dir_name="DialoGPT-"+model_size


torch.manual_seed(0)
np.random.seed(0)
EPSILON = 1e-10
example_sentence = "This is incredible! I love it, this is the best chicken I have ever had."
max_length_seq = 100


class Discriminator(torch.nn.Module):
    """Transformer encoder followed by a Classification Head"""

    def __init__(
            self,
            class_size,
            pretrained_model="gpt2-medium",
            cached_mode=False,
            device='cpu'
    ):
        super(Discriminator, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.encoder = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.embed_size = self.encoder.transformer.config.hidden_size
        self.classifier_head = ClassificationHead(
            class_size=class_size,
            embed_size=self.embed_size
        )
        self.cached_mode = cached_mode
        self.device = device

    def get_classifier(self):
        return self.classifier_head

    def train_custom(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier_head.train()

    def avg_representation(self, x):
        mask = x.ne(0).unsqueeze(2).repeat(
            1, 1, self.embed_size
        ).float().to(self.device).detach()
        hidden, _ = self.encoder.transformer(x)
        masked_hidden = hidden * mask
        avg_hidden = torch.sum(masked_hidden, dim=1) / (
                torch.sum(mask, dim=1).detach() + EPSILON
        )
        return avg_hidden

    def forward(self, x):
        if self.cached_mode:
            avg_hidden = x.to(self.device)
        else:
            avg_hidden = self.avg_representation(x.to(self.device))

        logits = self.classifier_head(avg_hidden)
        probs = F.log_softmax(logits, dim=-1)

        return probs


class Dataset(data.Dataset):
    def __init__(self, X, y):
        """Reads source and target sequences from txt files."""
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        data = {}
        data["X"] = self.X[index]
        data["y"] = self.y[index]
        return data


def collate_fn(data):
    def pad_sequences(sequences):
        lengths = [len(seq) for seq in sequences]

        padded_sequences = torch.zeros(
            len(sequences),
            max(lengths)
        ).long()  # padding value = 0

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_sequences[i, :end] = seq[:end]

        return padded_sequences, lengths

    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch, _ = pad_sequences(item_info["X"])
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch


def cached_collate_fn(data):
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    x_batch = torch.cat(item_info["X"], 0)
    y_batch = torch.tensor(item_info["y"], dtype=torch.long)

    return x_batch, y_batch

def train_epoch(data_loader, discriminator, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    samples_so_far = 0
    discriminator.train_custom()
    train_loss = 0
    for batch_idx, (input_t, target_t) in enumerate(data_loader):
        input_t, target_t = input_t.to(device), target_t.to(device)

        optimizer.zero_grad()

        output_t = discriminator(input_t)
        loss = F.nll_loss(output_t, target_t)
        # sum up batch loss
        train_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

        samples_so_far += len(input_t)

        #if batch_idx % log_interval == 0:
        #    print(
        #        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
        #            epoch + 1,
        #            samples_so_far, len(data_loader.dataset),
        #            100 * samples_so_far / len(data_loader.dataset), loss.item()
        #        )
        #    )
    train_loss /= (batch_idx+1)
    return train_loss


def evaluate_performance(data_loader, discriminator, device='cpu'):
    discriminator.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_t, target_t in data_loader:
            input_t, target_t = input_t.to(device), target_t.to(device)
            output_t = discriminator(input_t)
            # sum up batch loss
            test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
            # get the index of the max log-probability
            pred_t = output_t.argmax(dim=1, keepdim=True)
            correct += pred_t.eq(target_t.view_as(pred_t)).sum().item()

    test_loss /= len(data_loader.dataset)

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)".format(
            test_loss, correct, len(data_loader.dataset),
            100. * correct / len(data_loader.dataset)
        )
    )
    return test_loss, 100. * correct / len(data_loader.dataset)


def predict(input_sentence, model, classes, cached=False, device='cpu'):
    input_t = model.tokenizer.encode(input_sentence)
    input_t = torch.tensor([input_t], dtype=torch.long, device=device)
    if cached:
        input_t = model.avg_representation(input_t)

    log_probs = model(input_t).data.cpu().numpy().flatten().tolist()
    print("Input sentence:", input_sentence)
    print("Predictions:", ", ".join(
        "{}: {:.4f}".format(c, math.exp(log_prob)) for c, log_prob in
        zip(classes, log_probs)
    ))


def get_cached_data_loader(dataset, batch_size, discriminator,
                           shuffle=False, device='cpu'):
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)

    xs = []
    ys = []
    for batch_idx, (x, y) in enumerate(tqdm(data_loader, ascii=True)):
        with torch.no_grad():
            x = x.to(device)
            avg_rep = discriminator.avg_representation(x).cpu().detach()
            avg_rep_list = torch.unbind(avg_rep.unsqueeze(1))
            xs += avg_rep_list
            ys += y.cpu().numpy().tolist()

    data_loader = torch.utils.data.DataLoader(
        dataset=Dataset(xs, ys),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=cached_collate_fn)

    return data_loader

#parameters
dataset='SST'
#dataset_fp=descriminator_dataset_file
pretrained_model=model_dir_name
epochs=30
batch_size=64
log_interval=10000
save_model=True
cached=True
no_cuda=False


device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

print("Preprocessing {} dataset...".format(dataset))
idx2class = ["positive", "neutral","negative"]
class2idx = {c: i for i, c in enumerate(idx2class)}


discriminator = Discriminator(
    class_size=len(idx2class),
    pretrained_model=pretrained_model,
    cached_mode=cached,
    device=device
).to(device)

text = torchtext_data.Field()
label = torchtext_data.Field(sequential=False)
train_data, val_data, test_data = datasets.SST.splits(
    text,
    label,
    fine_grained=True,
    train_subtrees=True,
)

x = []
y = []
for i in trange(len(train_data), ascii=True):
    seq = TreebankWordDetokenizer().detokenize(
        vars(train_data[i])["text"]
    )
    seq = discriminator.tokenizer.encode(seq)
    seq = torch.tensor([50256] + seq , device=device, dtype=torch.long)
    x.append(seq)
    y.append(class2idx[vars(train_data[i])["label"].replace("very ","")])
train_dataset = Dataset(x, y)

test_x = []
test_y = []
for i in trange(len(test_data), ascii=True):
    seq = TreebankWordDetokenizer().detokenize(
        vars(test_data[i])["text"]
    )
    seq = discriminator.tokenizer.encode(seq)
    seq = torch.tensor([50256] + seq , device=device, dtype=torch.long)
    test_x.append(seq)
    test_y.append(class2idx[vars(test_data[i])["label"].replace("very ","")])
test_dataset = Dataset(test_x, test_y)



discriminator_meta = {
    "class_size": len(idx2class),
    "embed_size": discriminator.embed_size,
    "pretrained_model": pretrained_model,
    "class_vocab": class2idx,
    "default_class": 2,
}


print("Preprocessed {} data points".format(
    len(train_dataset) + len(test_dataset))
)



if cached:
    print("Building representation cache...")

    start = time.time()

    train_loader = get_cached_data_loader(
        train_dataset, batch_size, discriminator,
        shuffle=True, device=device
    )

    test_loader = get_cached_data_loader(
        test_dataset, batch_size, discriminator, device=device
    )

    end = time.time()
    print("Building representation cache took: {:.3f}s".format(end - start))

else:
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn)



if save_model:
    with open("{}_classifier_head_meta.json".format(dataset),
              "w") as meta_file:
        json.dump(discriminator_meta, meta_file)



optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)


train_losses=[]
test_losses=[]
test_accuracies=[]
for epoch in range(epochs):
    start = time.time()
    #print("\nEpoch", epoch + 1)

    train_loss=train_epoch(
        discriminator=discriminator,
        data_loader=train_loader,
        optimizer=optimizer,
        epoch=epoch,
        log_interval=log_interval,
        device=device
    )
    train_losses.append(train_loss)
    #print('Train Epoch: {}  Loss: {}'.format(epoch + 1,train_loss))
    test_loss, accuracy=evaluate_performance(
        data_loader=test_loader,
        discriminator=discriminator,
        device=device
    )
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)


    end = time.time()
    #print("Epoch took: {:.3f}s".format(end - start))

    #print("\nExample prediction")
    #predict(example_sentence, discriminator, idx2class,cached=cached, device=device)

    if save_model:
        # torch.save(discriminator.state_dict(),
        #           "{}_discriminator_{}.pt".format(
        #               args.dataset, epoch + 1
        #               ))
        torch.save(discriminator.get_classifier().state_dict(),
                   "{}_classifier_head_epoch_{}.pt".format(dataset,
                                                           epoch + 1))



# libraries and data
plt.rcParams.update({'font.size': 14})
df=pd.DataFrame({'xvalues': range(1,epochs+1), 'train_losses': train_losses, 'test_losses': test_losses, 'test_accuracies': test_accuracies })

# plot
plt.figure(figsize=(7,5))
plt.plot( 'xvalues', 'train_losses', data=df, label='train loss')
plt.plot( 'xvalues', 'test_losses', data=df, label='test loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc="upper right")
plt.show()



# plot
plt.figure(figsize=(7,5))
plt.rcParams.update({'font.size': 14})
plt.plot( 'xvalues', 'test_accuracies', data=df, label='test accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc="upper left")
plt.show()


data_loader=test_loader
discriminator.eval()
test_loss = 0
target_ts=[]
pred_ts=[]
with torch.no_grad():
    for input_t, target_t in data_loader:
        target_ts.extend(target_t.numpy())
        input_t, target_t = input_t.to(device), target_t.to(device)
        output_t = discriminator(input_t)
        # sum up batch loss
        test_loss += F.nll_loss(output_t, target_t, reduction="sum").item()
        # get the index of the max log-probability
        pred_t = output_t.argmax(dim=1, keepdim=True)
        pred_ts.extend(pred_t.view_as(target_t).cpu().numpy())


#Plotting the classifiaction matrix
plt.rcParams.update({'font.size': 14})
conf_mat = confusion_matrix(target_ts, pred_ts)
fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=idx2class, yticklabels=idx2class, cmap=sns.color_palette("Blues"))
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


shutil.copyfile('SST_classifier_head_epoch_'+epochs+'.pt', 'SST_classifier_head.pt')
