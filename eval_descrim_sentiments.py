# -*- coding: utf-8 -*-

from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer

from pplm_eval import generate_text_pplm
import torch
import numpy as np
import json
from typing import List, Optional, Tuple, Union
from pplm_classification_head import ClassificationHead
from transformers.file_utils import cached_path
import time
import json

num_samples=5
eval_dataset_type="preference"
eval_dataset_files={"preference": "datasets/Personal Preference Questions Dataset.txt", "opinion": "datasets/Opinion Questions Dataset.txt"}
model_size = "medium"
pretrained_model="DialoGPT-"+model_size
discrim_meta_file="SST_classifier_head_meta.json"
discrim_weights="SST_classifier_head.pt"

idx2class = ["positive", "neutral","negative"]
class2idx = {c: i for i, c in enumerate(idx2class)}




with open(eval_dataset_files[eval_dataset_type],'r') as f:
    f_content = f.readlines()
queries=[x.strip() for x in f_content]

loss_type = 2
def get_classifier(
        discrim_meta: Optional[dict],
        class_label: Union[str, int],
        device: str
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if discrim_meta is None:
        return None, None

    params = discrim_meta
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]

    else:
        label_id = params["default_class"]

    return classifier, label_id

# set Random seed
seed=0
torch.manual_seed(seed)
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained model
model = GPT2LMHeadModel.from_pretrained(pretrained_model,output_hidden_states=True)

# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

model.to(device)
model.eval()

# Freeze GPT-2 weights
for param in model.parameters():
    param.requires_grad = False

with open(discrim_meta_file, 'r') as f:
    discrim_meta = json.load(f)
discrim_meta['path'] = discrim_weights

assert discrim_meta['pretrained_model']==pretrained_model


descr={}
for class_label in idx2class:
  classifier, class_id = get_classifier(
    discrim_meta,
    class_label,
    device
  )
  descr[class_label]=(classifier, class_id)

responses=[]
for query_num,cond_text in enumerate(queries):
  print('Query {} started'.format(query_num+1))
  response={"query": cond_text}
  tokenized_cond_text = tokenizer.encode(tokenizer.bos_token + cond_text+tokenizer.bos_token,add_special_tokens=False)
  unpert_gen_tok_text, _, _ = generate_text_pplm(
      model=model,
      tokenizer=tokenizer,
      context=tokenized_cond_text,
      device=device,
      length=50,
      sample=True,
      perturb=False
  )
  response["unperturbed"]=tokenizer.decode(unpert_gen_tok_text.tolist()[0])
  #if device == 'cuda':
  #  torch.cuda.empty_cache()

  for class_label in idx2class:
    response[class_label]={'utterances':[],'losses':[]}
    classifier, class_id=descr[class_label]
    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    for i in range(num_samples):
      pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
          model=model,
          tokenizer=tokenizer,
          context=tokenized_cond_text,
          device=device,
          perturb=True,
          classifier=classifier,
          class_label=class_id,
          loss_type=loss_type,
          length=30,
          stepsize=0.2,
          temperature=1.0,
          top_k=10,
          sample=True,
          num_iterations=3,
          grad_length=10000,
          horizon_length=1,
          window_length=10,
          decay=False,
          gamma=1,
          gm_scale=0.95,
          kl_scale=0.01
      )
      pert_gen_tok_texts.append(pert_gen_tok_text)
      if classifier is not None:
        discrim_losses.append(discrim_loss.data.cpu().numpy().tolist())
      losses_in_time.append(loss_in_time)
      if pert_gen_tok_text is not None:
        response[class_label]['utterances'].append(tokenizer.decode(pert_gen_tok_text.tolist()[0]))
      else:
        response[class_label]['utterances'].append('')
    #if device == 'cuda':
    #  torch.cuda.empty_cache()
    response[class_label]['losses']=discrim_losses
  responses.append(response)

with open('response-'+eval_dataset_type+'-sentiment.json','w') as f:
  json.dump(responses,f)
