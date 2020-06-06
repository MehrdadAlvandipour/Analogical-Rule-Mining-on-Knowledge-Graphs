import os
import sys
import datetime
import time
import re
import shutil
from shutil import copyfile
import shlex,subprocess


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from tqdm.notebook import trange
from itertools import combinations
import itertools
from . import xUtils


class xSimilarity(dict):
  def __getitem__(self, item):
      try:
        return dict.__getitem__(self, item) % self
      except TypeError: #some values in dict are not str and % fails.
        return dict.__getitem__(self, item)
  #def __repr__(self):
  #  print("current_dataset: " + self.current_dataset)
  #  print("training_data_path: " + self.training_data_path)
  #  print("base_data_path: " + self.base_data_path)
  #  return dict.__repr__(self) % self

  def __init__(self,root_folder=None,
               backup='/content/drive/My Drive/Colab Notebooks/backup'):
    
    if root_folder == None:
      root_folder = os.getcwd()
    self._openKE_installed = False 

    self._current_dataset = ''
    self._training_data_path = ''
    self._base_data_path = ''

    self._checkpoint_file = ''
    self._ckpt_logfile = ''

    self.openke_objects = {'train_dataloader': None,
                           'test_dataloader': None,
                           'distmult': None,
                           'model': None,
                           'trainer': None,
                           'tester': None}

    self['root'] = root_folder.rstrip('/. ')
    self['openKE'] = '%(root)s/OpenKE'
    self['cloud_datasets'] = ''
    self['local_datasets'] = '%(root)s/datasets'
    self['backup'] = backup.rstrip('/. ')
    self['benchmarks_folder'] = '%(openKE)s/benchmarks'
    self['benchmarks_dict'] = {}
    self['enriched_files'] = []
    self['checkpoints'] = '%(root)s/checkpoints'
    self['rules'] = '%(root)s/rules'
    self['evaluation'] = '%(root)s/evaluation'
    
    if not os.path.exists(self['checkpoints']):
      os.mkdir(self['checkpoints'])
    if not os.path.exists(self['rules']):
        os.mkdir(self['rules'])
    if not os.path.exists(self['evaluation']):
        os.mkdir(self['evaluation'])

    
  
  def _get_size(self,data_folder,key):
    keys = {'train_size':'train2id.txt',
          'test_size': 'test2id.txt',
          'ent_total': 'entity2id.txt',
          'rel_total': 'relation2id.txt'}
    file = os.path.join(data_folder,keys[key])
    f = open(file)
    value = int(f.readline().rstrip())
    f.close()
    return value

  def _create_benchmarks_dict(self):
    benchmarks = {}
    for d in os.listdir(self['benchmarks_folder']):
      if not (os.path.isdir(  os.path.join(self['benchmarks_folder'],d) )):
        continue
      data_folder = os.path.join(self['benchmarks_folder'],d)
      train_size = self._get_size(data_folder,'train_size')
      test_size = self._get_size(data_folder,'test_size')
      ent_total = self._get_size(data_folder,'ent_total')
      rel_total = self._get_size(data_folder,'rel_total')
      benchmarks[d] = {  'train_size': train_size, 
                            'test_size': test_size,
                            'ent_total': ent_total,
                            'rel_total': rel_total}
    self['benchmarks_dict'] =  benchmarks  
  

  def _load_enriched_files(self):
    self['enriched_files'] = []
    for file in os.listdir(self.base_data_path):
      if ('train_' in file):
        self['enriched_files'].append(file)
     



# Dynamic properties '--------------------'  
  @property
  def openKE_installed(self):
    return self._openKE_installed
  @openKE_installed.setter
  def openKE_installed(self,new_value):
    self._openKE_installed = new_value
    if new_value:
      self._create_benchmarks_dict()
      if self.current_dataset:
        self._training_data_path = (self['benchmarks_folder'] 
                                 + '/' 
                                 + self._current_dataset  
                                 + '/')
    else:
      self['benchmarks_dict'] = {}


  
  @property
  def current_dataset(self):
    return self._current_dataset  
  @current_dataset.setter
  def current_dataset(self,new_value):
    '''Set the current dataset and manage the related attributes.
    Other attributes include: training_data_path, base_data_path, openKE_installed

    '''
    if (not self.openKE_installed) and os.path.isdir(self['openKE']):
      self.openKE_installed = True

    if new_value not in os.listdir(self['local_datasets']) and new_value not in self['benchmarks_dict']:
      print(new_value + ' does not exists anywhere')
      return

    self._current_dataset = new_value 
    self._base_data_path =  os.path.join(self['local_datasets'],self.current_dataset)
    self._load_enriched_files()
    if self.openKE_installed:
      self._training_data_path = (self['benchmarks_folder'] 
                                 + '/' 
                                 + self._current_dataset  
                                 + '/')
    #else:
    #  print('OpenKE not installed. training_data_path attribute is not set.')

  


 # Read-only properties '--------------------'
  @property
  def training_data_path(self):
    return self._training_data_path

  @property
  def base_data_path(self):
    return self._base_data_path


  @property
  def checkpoint_file(self):
    self._checkpoint_file = (self['checkpoints'] +
                             '/' +
                             'dist_' +
                             self.current_dataset + 
                             '.ckpt')
    return self._checkpoint_file
  
  @property
  def ckpt_logfile(self):
    self._ckpt_logfile = (self['checkpoints'] +
                          '/' +
                          'ckpt_' +
                          self.current_dataset + 
                          '_log.txt')
    return self._ckpt_logfile  
  
  
 

 # Public Methods '--------------------'

  def install_openke(self,source):
    if self.openKE_installed and os.path.isdir(self['openKE']):
      print('Already installed')
      return None
     
    os.chdir(self['root'])
    if source == 'git':
      print('Installing from gitHub')
      cmd = "git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE"
      os.system(cmd)
    else:
      src = os.path.join(self['backup'],'OpenKE')
      dst = os.path.join(self['root'],'OpenKE')
      print('Copying from ' + src)
      shutil.copytree(src,dst)
    

    return_code = xUtils.run_proc('bash make.sh',cwd = os.path.join(self['openKE'],'openke'))[0]

    print(return_code)
    if return_code == 0:
      self.openKE_installed = True
      print('Successfuly installed OpenKE.')
      self._create_benchmarks_dict()
    else:
      print("Something went wrong.")
      raise
  
  
  
  def log_test_results(self, test_results, 
                       checkpoint_file=None):
    if checkpoint_file == None:
      checkpoint_file = self.checkpoint_file

    filename = self.ckpt_logfile
    if os.path.exists(filename):
      append_write = 'a' # append if already exists
    else:
      append_write = 'w' # make a new file if not

    with open(filename,append_write) as file:
      line = ("\n" + 
              checkpoint_file[checkpoint_file.rfind('/')+1:] 
              + "\t" + str(test_results))
      out = file.write(line)
      print(str(out) + ' characters added to:')
      print(filename)

  def backup_file(self,src,time_stamp = True):
    '''Note:
    It puts the time stamp after the extension.
    '''
    if not os.path.isfile(src):
      print('You can only backup files with this command.')
      return -1
    if not os.path.isdir(self['backup']):
      print("Backup directory failure. Check self['backup']")
      return -1

    filename = src[src.rfind('/')+1:]
    if time_stamp:
      filename = filename + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    dst = os.path.join(self['backup'],filename)
    copyfile(src,dst)
    print('Backup saved at: ', dst)



 ## OpenKE methods '-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~'

  def openKE_load(self):
    if not self.openKE_installed:
      print('OpenKE not installed.')
    if self.training_data_path == '' or self.current_dataset == '':
      print('select your current data set first.')
      raise
    
    sys.path.append(self['openKE'])
    import openke
    from openke.module.model import DistMult
    from openke.module.loss import SigmoidLoss
    from openke.module.strategy import NegativeSampling
    from openke.data import TrainDataLoader, TestDataLoader 
    from openke.config import Trainer, Tester

    
    self.openke_objects['train_dataloader'] = TrainDataLoader(
      in_path = self.training_data_path, 
      batch_size = 2048,
      threads = 8,
      sampling_mode = "cross", 
      bern_flag = 0, 
      filter_flag = 1, 
      neg_ent = 64,
      neg_rel = 0
    )
    self.openke_objects['test_dataloader'] = TestDataLoader(self.training_data_path, "link")
    self.openke_objects['distmult'] = DistMult(
      ent_tot = self.openke_objects['train_dataloader'].get_ent_tot(),
      rel_tot = self.openke_objects['train_dataloader'].get_rel_tot(),
      dim = 128,
      margin = 200.0,
      epsilon = 2.0
    )
    self.openke_objects['model'] = NegativeSampling(
      model = self.openke_objects['distmult'], 
      loss = SigmoidLoss(adv_temperature = 0.5),
      batch_size = self.openke_objects['train_dataloader'].get_batch_size(), 
      l3_regul_rate = 0.000005
    )

  def openKE_train(self,train_times = 400, alpha = 0.002, use_gpu = True, opt_method = "adam"):
    #os.chdir(self['openKE'])
    from openke.config import Trainer
    self.openke_objects['trainer'] = Trainer(model = self.openke_objects['model'], data_loader = self.openke_objects['train_dataloader'], train_times = 400, alpha = 0.002, use_gpu = True, opt_method = "adam")
    self.openke_objects['trainer'].run()
    self.openke_objects['distmult'].save_checkpoint(self.checkpoint_file)
    print('checkpoint saved at:')
    print(self.checkpoint_file)
    self.backup_file(self.checkpoint_file) 

  def openKE_test(self):
    from openke.config import Tester
    self.openke_objects['distmult'].load_checkpoint(self.checkpoint_file)
    self.openke_objects['tester'] = Tester(model = self.openke_objects['distmult'], data_loader = self.openke_objects['test_dataloader'], use_gpu = True)
    test_results = self.openke_objects['tester'].run_link_prediction(type_constrain = False)
    self.log_test_results(test_results)
  
  def openKE_load_tester_from_file(self,file):
    print('Make sure ckpt file matches with the current dataset:')
    print(self.current_dataset + ' at ')
    print(self.training_data_path)
    self.openKE_load()
    from openke.config import Tester
    self.openke_objects['distmult'].load_checkpoint(file)
    self.openke_objects['tester'] = Tester(model = self.openke_objects['distmult'], data_loader = self.openke_objects['test_dataloader'], use_gpu = True)
    

  def cosD_scores(self):
    tester = self.openke_objects['tester']
    ent_total = self['benchmarks_dict'][self.current_dataset]['ent_total']
    
    import torch
    indices = torch.cuda.LongTensor(range(ent_total))
    ents = tester.model.ent_embeddings(indices)
    ents_mag = torch.norm(ents,dim=1)
    ents_normalized = torch.div(ents,ents_mag.view((ent_total,1)))

    

    chunk_size =  2**11  
    tot = ent_total * (ent_total - 1)/2
    step_total = np.ceil(tot/chunk_size)
    print( 'Total steps: ' + str(step_total))

    h_t_pairs = combinations(range(ent_total),2)
    all_scores = np.empty([int(step_total),int(chunk_size)])

    for c in trange(int(step_total)):
      p = itertools.islice(h_t_pairs,0,chunk_size)
      h,t = zip(*list(p))
      h = np.array(h)
      t= np.array(t)

      ent_h = ents_normalized[h]
      ent_t = ents_normalized[t]
      res = torch.sum(torch.mul(ent_h,ent_t),1)

      try:
        all_scores[c,:] = res.cpu().data.numpy()
      except ValueError: # Last batch may not be full size
        _ = res.cpu().data.numpy()
        all_scores[c,:_.size] = _

    all_scores = all_scores.reshape(-1)[:int(tot)]

    # We only keep top k scores where k is the training size of the benchmark.
    train_size = self['benchmarks_dict'][self.current_dataset]['train_size']
    
    # higher is better for cosine score, so partition and keep from the end.
    crop_at = -train_size 
    ind = np.argpartition(all_scores,crop_at)[crop_at:]
    all_scores = all_scores[ind]

    # Keep the h,t pairs corresponding to ind
    sel = np.zeros(int(tot))
    sel[ind] = 1
    tab =  itertools.compress(combinations(range(ent_total),2) , sel)
    h,t = zip(*list(tab))

    file = os.path.join( x['root'], f'cosD_{self.current_dataset}.npz')
    np.savez(file, head=h, tail=t, score=all_scores)
    print('\n Scores saved at: ' + file)

    self.backup_file(file)

    df = pd.DataFrame(data={'head':h, 'tail':t, 'score': all_scores})
    return df

  def dist_avg_scores(self):
    '''Note:
    Requires openKE tester object at: self.openke_objects['tester'] 

    '''
    tester = self.openke_objects['tester']
    ent_total = tester.model.ent_tot
    rel_total = tester.model.rel_tot
    assert ((self['benchmarks_dict'][self.current_dataset]['ent_total'],
     self['benchmarks_dict'][self.current_dataset]['rel_total']) == 
     (ent_total, rel_total))
    
    chunk_size =  2**11  
    tot = ent_total * (ent_total - 1)/2
    step_total = np.ceil(tot/chunk_size)
    print( 'Total steps: ' + str(step_total))

    r = np.array(range(rel_total))
    h_t_pairs = combinations(range(ent_total),2)
    all_scores = np.zeros([int(step_total),int(chunk_size)])

    for c in trange(int(step_total)):
      p = itertools.islice(h_t_pairs,0,chunk_size)
      h,t = zip(*list(p))
      h = np.array(h)
      t= np.array(t)
      
      r_t = np.tile(r,h.size)
      
      data = {'batch_h': h.repeat(rel_total) , 'batch_t':t.repeat(rel_total), 'batch_r':r_t, 'mode': 'head_batch'}
      res = tester.test_one_step(data)
      ind = res < 0
      dis_avg = np.divide(np.sum(np.multiply(res,ind).reshape(-1,rel_total),1), np.sum(ind.reshape(-1,rel_total),1)+ 1e-9 ) 
      
      try:
        all_scores[c,:] = dis_avg
      except ValueError: # Last batch may not be full size
        all_scores[c,:dis_avg.size] = dis_avg
    
    all_scores = all_scores.reshape(-1)[:int(tot)]
    
    train_size = self['benchmarks_dict'][self.current_dataset]['train_size']
    crop_at = train_size
    ind = np.argpartition(all_scores,crop_at)[:crop_at]
    all_scores = all_scores[ind]


    sel = np.zeros(int(tot))
    sel[ind] = 1
    tab =  itertools.compress(combinations(range(ent_total),2) , sel)
    h,t = zip(*list(tab))

    file = os.path.join( self['root'], f'dist_{self.current_dataset}.npz')
    np.savez(file, head=h, tail=t, score=all_scores)
    print('\n Scores saved at: ' + file)
    self.backup_file(file)

    df = pd.DataFrame(data={'head':h, 'tail':t, 'score': all_scores})
    return df


  def load_from_npz(self,file=None):
    if file==None:
      file = os.path.join( self['root'], f'dist_{self.current_dataset}.npz')
    data = np.load(file)
    h = data['head']
    t = data['tail']
    score = data['score']
    df = pd.DataFrame(data={'head':h, 'tail':t, 'score': score})
    return df

    
 ## Filtering df '-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~'

  def append_train(self,input_df,method_name):
      """ Appends the input data frame to a copy of train.txt.
      
      input_df: --pd.DataFrame: has two columns 'head', and 'tail' containing
      the integer ids for heads and tails of similar tuples.
      method_name: --str: To be used as a suffix for the output -> train_{new_name}_{time}.txt 
      """
      new_name = method_name + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
      new_name = 'train_' + new_name + '.txt'
      dest = os.path.join(self['local_datasets'],self.current_dataset, new_name)

      while os.path.isfile(dest): #check if the file already exists, if so ask for a new name
          new_name = input("File already exists. Give another name: ")
          new_name = 'train_' + new_name + '.txt'
          dest = os.path.join(self['local_datasets'],self.current_dataset, new_name)
      
      heads = list(input_df['head']) 
      tails = list(input_df['tail']) 

      # Translate int ids to /mid
      ents = pd.read_csv(os.path.join(self.training_data_path,"entity2id.txt"),sep = '\t',header=None, skiprows=[0],usecols=[0]) # first row is lineTot
      heads_mid = list(ents.iloc[heads,0]) 
      rels_mid = ['/similar_to']*len(heads)
      tails_mid = list(ents.iloc[tails,0]) 

      d = {'head': heads_mid , 'relation': rels_mid, 'tail':tails_mid}
      df = pd.DataFrame(data=d)

      
      copyfile( os.path.join(self.base_data_path,'train.txt'), dest)
      df.to_csv(dest, mode='a', header=False,index=False, sep='\t')
      return new_name


  def best_scores(self,df,percent, asc=1):
    """filters the input <df> based on its 'score' column, 
    keeping only top <percent>."""
    if percent > 1:
      percent = percent / 100
    if percent ==1:
      return df
    
    crop_index = int(percent*len(df))
    if asc == 1:
      df_filtered = df.iloc[np.argpartition(df['score'],crop_index)[:crop_index]]
    elif asc == -1:
      crop_index = -1 * crop_index
      df_filtered = df.iloc[np.argpartition(df['score'],crop_index)[crop_index:]]
    return df_filtered

  def filter_and_append(self,df,percent, method):
    method_options = {'distMult','cosD','Sub'}
    if method not in method_options:
      print('method argument must be one of: ', method_options)
      raise
    
    # Filter based on the method.
    if method == 'distMult':
      new_df = self.best_scores(df,percent)
    else:
      new_df = self.best_scores(df,percent,asc=-1)
    
    # append to training file with a suffix
    suffix = f'{method}_' + str(percent) + 'p_'
    enriched_file_name = self.append_train(new_df,suffix)

    # Keep track of enriched files
    self['enriched_files'].append(enriched_file_name)

 




 ## Subject Similarity '-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~'

  def sub_scores(self):
    ent_total = self['benchmarks_dict'][self.current_dataset]['ent_total']    # From the first line of ent2id.txt
    training_file = os.path.join(self.training_data_path, 'train2id.txt') 
    s = [ [] for _ in range(ent_total)] # A list of lists(sets). Will contain all sets S_{e_i}.


    # Create the sets
    f = open(training_file,'r')
    num_lines = int(f.readline())    # First line of train2id is the number of triplets
    assert num_lines == self['benchmarks_dict'][self.current_dataset]['train_size']
    for i in trange(num_lines):
      l = f.readline()
      l = l.split()
      try:
        s[int(l[0])] += [ ( int(l[1]) , int(l[2]) ) ]
      except:
        print("something went wrong at triple" + str(i))
        raise
    f.close()


    # Compute the scores
    h = []
    t = []
    sc = []
    with tqdm(total=int(ent_total*(ent_total-1)/2)) as pbar:
      for i,j in combinations(range(ent_total),2):
        score = len( set(s[i]) & set(s[j]) )
        if score:
          h.append(i)
          t.append(j)
          sc.append(score)
        pbar.update(1)

    

    d = {'head':h , 'tail':t, 'score': sc}
    df = pd.DataFrame(data=d)

    # We only keep a df with the same size as the training
    df = df.iloc[np.argpartition(df['score'],-num_lines)[-num_lines:]]

    file = os.path.join( self['root'], f'Sub_{self.current_dataset}.npz')
    np.savez(file, head=df['head'], tail=df['tail'], score=df['score'])
    print('\n Dataframe saved at: ' + file)
    self.backup_file(file)
    

    return df


 ## AMIE '-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~'


  def clean_amie_output(self,path):
    """
    Warning: this function overwrites the file in path
    """
    with open(path, 'r') as f:
      f_contents = f.readlines()
    f_contents = f_contents[13:]
    with open(path, 'w') as f:
      for line in f_contents:
        if max(pattern in line for pattern in ['Mining done', 'Total time', 'rules mined', 'countPairs']):  #re.search(r'Mining done', line):
          continue
        else:
          if not (line == '\n'):
            f.write(line)
    #print('Rules at %s file cleaned.' % path)

  def eval_frame(self, file, test_len):
      
      # Open file
      f = open(file)
      
      # Hits counter
      hits = 0
      
      # Loop though all facts in KB
      for x in range(test_len):

          # Read line
          fact = f.readline()
          fact = fact.split(' ')
          if fact != ['']:
              # Get target head and tail
              head_target = fact[0]
              tail_target = fact[2][:-1]


              # Get head predictions
              headpreds = f.readline()
              headpreds = headpreds.split(' ')
              headpreds = headpreds[1].split('\t')
              headpreds.pop()

              # Get tail predictions
              tailpreds = f.readline()
              tailpreds = tailpreds.split(' ')
              tailpreds = tailpreds[1].split('\t')
              tailpreds.pop()


              if (head_target in headpreds) and (tail_target in tailpreds):
                  if (len(headpreds) < 10) and (len(tailpreds) < 10):
                      hits+=1
          else:
              print('miss')
              raise 
                  
      return hits/(test_len)

  def AMIE_mine_baseline(self,max_heap_size=16,minhc = 0.25, mins=50, minis=0):
    train_add =  os.path.join(self.base_data_path,'train.txt')
    test_add = os.path.join(self.base_data_path,'test.txt')
    valid_add = os.path.join(self.base_data_path,'valid.txt')

    rules_add = os.path.join(self['rules'],f"{self.current_dataset}_baseline_rules.txt")
    eval_add = os.path.join(self['evaluation'],f"{self.current_dataset}_baseline_rules_eval.txt")

    AMIE_local_location = os.path.join(self['root'],"AMIE")

    AMIE_plus = (f"java -XX:-UseGCOverheadLimit -Xmx{max_heap_size}g -jar {AMIE_local_location}/amie_plus.jar "
    f"-minhc {minhc} -mins {mins} -minis {minis} " 
    f"{train_add} > {rules_add}")

    Apply_AMIE_RULES = (f'java -jar {AMIE_local_location}/ApplyAMIERules.jar {rules_add}' 
                        f' {train_add} {test_add} {valid_add}'
                        f' {eval_add}')


    self.run_AMIE_and_backup_rules(AMIE_plus,Apply_AMIE_RULES,test_add,rules_add,eval_add)



  def AMIE_mine_enriched(self,max_heap_size=16,minhc = 0.25, mins=50, minis=0):
    '''Mine rules from all the enriched files in the current dataset. The file names can be 
    accessed with key: ['enriched_files']
    Enriched file names starts with 'train_'.
    '''

    for enriched_file in self['enriched_files']:
      train_add =  os.path.join(self.base_data_path,enriched_file)
      test_add = os.path.join(self.base_data_path,'test.txt')
      valid_add = os.path.join(self.base_data_path,'valid.txt')

      rules_add = os.path.join(self['rules'],self.current_dataset + '_rules' + enriched_file[10:])
      eval_add = os.path.join(self['evaluation'],self.current_dataset + '_eval' + enriched_file[10:])

      AMIE_local_location = os.path.join(self['root'],"AMIE")

      AMIE_plus = (f"java -XX:-UseGCOverheadLimit -Xmx{max_heap_size}g -jar {AMIE_local_location}/amie_plus.jar "
      f"-minhc {minhc} -mins {mins} -minis {minis} " 
      f"{train_add} > {rules_add}")


      Apply_AMIE_RULES = (f'java -jar {AMIE_local_location}/ApplyAMIERules.jar {rules_add}' 
                          f' {train_add} {test_add} {valid_add}'
                          f' {eval_add}')

      self.run_AMIE_and_backup_rules(AMIE_plus,Apply_AMIE_RULES,test_add,rules_add,eval_add)
      print('\n')


  def run_AMIE_and_backup_rules(self,AMIE_plus,Apply_AMIE_RULES,test_add,rules_add,eval_add):
    
    print('running AMIE_plus')
    rule_file = open(rules_add,'w')
    AMIE_proc = subprocess.run(AMIE_plus,stdout=rule_file, stderr=subprocess.DEVNULL,shell=True)
    time.sleep(1)
    rule_file.close()
    if AMIE_proc.returncode != 0:
      print(f'AMIE_plus failed. Check errors in {rules_add}')
      return

    copyfile(rules_add,rules_add+'.backup')
    self.clean_amie_output(rules_add)
    time.sleep(1)

    print('running Apply_AMIE_RULES')
    eval_file = open(eval_add,'w')
    AMIE_proc = subprocess.run(Apply_AMIE_RULES,stdout=eval_file, stderr=subprocess.STDOUT,shell=True)
    time.sleep(1)
    eval_file.close()
    if AMIE_proc.returncode != 0:
      print(f'Apply_AMIE_RULES failed. Check errors in {eval_add}')
      return

    test_size = xUtils.file_len(test_add)
    print(eval_add)
    print('Hits@10: ' + str(self.eval_frame(eval_add, test_size)))

