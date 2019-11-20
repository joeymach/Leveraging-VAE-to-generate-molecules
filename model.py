#runs on google colab

#update tensorfzlow to 2.0  
!pip uninstall tensorflow #runtime = none

!pip uninstall tensorflow-gpu #runtime = gpu  
!pip install tensorflow-gpu==2.0.0-alpha0

!pip show tensorflow
!pip install --upgrade Tensorflow
import tensorflow as tf
print(tf.__version__)

!pip show tensorflow

#download deepchem library
!wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!time bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local
!conda install -y -c deepchem -c rdkit -c conda-forge -c omnia deepchem-gpu=2.1.0 python=3.6

import sys
if sys.version_info[0] >= 3:
    sys.path.append('/usr/local/lib/python3.6/site-packages/')
sys.path


#import training dataset
import deepchem as dc
tasks, datasets, transformers = dc.molnet.load_muv() 
train_dataset, valid_dataset, test_dataset = datasets 
train_smiles = train_dataset.ids

tokens = set()
for s in train_smiles:
  tokens = tokens.union(set(s))
tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles)

#training
from deepchem.models.tensorgraph.optimizers import Adam, ExponentialDecay
from deepchem.models.tensorgraph.models.seqtoseq import AspuruGuzikAutoEncoder
#the encoder is a CNN and the decoder is a GRU
model = AspuruGuzikAutoEncoder(tokens, max_length, model_dir='vae')

batches_per_epoch = len(train_smiles)/model.batch_size
learning_rate = ExponentialDecay(0.001, 0.95, batches_per_epoch)
model.set_optimizer(Adam(learning_rate=learning_rate))

def generate_sequences(epochs): 
  for i in range(epochs):
    for s in train_smiles: 
      yield (s, s)
model.summary()
model.fit_sequences(generate_sequences(1))

#check that the molecules are valid
import numpy as np
from rdkit import Chem
predictions = model.predict_from_embeddings(np.random.normal(size=(1000,196))) 
molecules = []
for p in predictions:
  smiles = ''.join(p) morning 
  if Chem.MolFromSmiles(smiles) is not None:
    molecules.append(smiles) 

for m in molecules:
  print(m)

smiles_list = [Chem.MolFromSmiles(x) for x in molecules]
print(sorted([x.GetNumAtoms() for x in smiles_list]))

good_mol_list = [x for x in smiles_list if x.GetNumAtoms() > 10 
                 and x.GetNumAtoms() < 50]
print(len(good_mol_list))

#obtain QED(drug-likeness) - drop all molecules with QED below 0.5
from rdkit.Chem import QED
qed_list = [QED.qed(x) for x in good_mol_list] 
final_mol_list = [(a,b) for a,b in
  zip(good_mol_list,qed_list) if b > 0.5]

for i in final_mol_list:
  print(i)

from rdkit.Chem.Draw import MolsToGridImage

#printing out the drawings of generated molecules 
MolsToGridImage([x[0] for x in final_mol_list], molsPerRow=3,useSVG=True, 
                subImgSize=(250, 250),
                legends=[f"{x[1]:.2f}" for x in final_mol_list])
