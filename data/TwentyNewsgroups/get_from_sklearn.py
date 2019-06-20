#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''

@author: Steffen Remus (@remstef)
'''

import sklearn.datasets
import pandas
import os

def save_as_txt_files(subset):
  # get the raw data and remove headers, footers, and quotes
  messages = sklearn.datasets.fetch_20newsgroups(subset=subset, remove=('headers','footers','quotes'), shuffle=False, random_state=42)
  samples = pandas.DataFrame(data=messages.data, columns=['rawdata'])
  samples['labelid'] = messages.target
  samples['filename'] = messages.filenames
  samples['basename'] = samples.filename.apply(os.path.basename)
  samples['label'] = samples.labelid.apply(lambda lid: messages.target_names[lid])
  samples.rawdata = samples.rawdata.apply(str.strip)
  samples = samples[samples.rawdata.apply(len) > 0]
  labels = samples.label.unique()

  for label in labels:
    d = os.path.join(subset, label)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, 'docs.txt'), 'wt') as f:
      samples_with_label = samples[samples['label']==label]
      samples_with_label.apply(lambda r: print(r.rawdata.replace('\r','').replace('\n','\\n'), file=f), axis=1)

save_as_txt_files('test')
save_as_txt_files('train')



