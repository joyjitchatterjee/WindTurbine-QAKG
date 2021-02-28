# -*- coding: utf-8 -*-

import pandas as pd

subsystem_names = ["Transformer","Drive Train","Foundation & Concrete Section","Converter","Pitch System","Power Cabinet",\
                   "Ground Line & Lightning Protection","Generator","Electric, Sensor &Control","Hydraulic System",\
                   "Park Brake","Yaw System","Blades"] #For replacing <subsys-name> elements


scada_des_file = pd.read_csv("/data/scadadescriptions.csv", header=0)
scada_description = list(scada_des_file.Description) #For replacing <scadadescription> elements e.g. Pitch Angle Mean Value etc. (full described name)

functionalgroup_names = ["No fault","Partial Performance-Degraded","Pitch System Interface Alarms",\
                         "Gearbox","Pitch System EFC Monitoring","PCS",\
                         "MVTR","Yaw Brake","Hydraulic System","Yaw",\
                         "Wind Condition Alarms","Pitch","IPR","Test"] #For replacing <fng-name elements e.g. Partial Performance-Degraded, Pitch etc.


scada_names_file = pd.read_csv("/data/scadanames.csv", header=0)
feature_labels = list(scada_names_file.Name) #For replacing <scadaname> elements e.g. Pitch_Deg_Mean (shortened name/data labels for features)


feature_numbers = list(range(0, 102)) #for replacing <scadafeatureno> elements e.g. 0 (1st feature), 1 (2nd feature etc. ) till 101 (102th feature) based on 0 indexing

# Using list comprehension to convert to string-type list 
feature_numbers = [str(x) for x in feature_numbers] 
    

alarm_numbers = list(range(901, 927)) # for replacing <alarmno> elements *alarm no. 901-926 in the KG

# Using list comprehension to convert to string-type list 
alarm_numbers = [str(x) for x in alarm_numbers] 
    
alarmdes_dict = {
    'Turbine operating normally': 0,
    'Partial Performance - Degraded': 1,
    'Pitch Heartbeat Error': 2,
    '(DEMOTED) Gearbox Filter Manifold Pressure 1 Shutdown': 3,
    'Pitch System Fatal Error': 4,
    'Blade 1 too slow to respond': 5,
    'PcsTrip': 6,
    'MVTR Input Air Temp Shutdown':7,
    '(DEMOTED) Yaw Brake 2 Under Pressure Full Brake': 8,
    'HPU 2 Pump Active For Too Long' : 9,
    'Yaw Error > Max Start Yaw Error': 10,
    'Wind Speed Above Max Start' : 11,
    'Blade 3 too slow to respond' : 12,
    'Sub Pitch Priv Fatal Error has occurred more than 3 times in 86400 Seconds' : 13,
    '(DEMOTED) Yaw Brake 2 Under Pressure Reduced Brake' : 14,
    'PitchBladeAtFineLimitSwitch23' : 15,
    '(DEMOTED) Yaw Brake 1 Over Pressure Reduced Brake' : 16,
    '(DEMOTED) Yaw Hydraulic Pressure Diff Too Large' : 17,
    'PcsFaulted' : 18,
    'Yaw Move In Wrong Direction' : 19,
    '(DEMOTED) Gearbox Oil Tank 2 Level Shutdown' : 20,
    'Sub Pitch Priv Critical Error has occurred more than 3 times in 86400 Seconds' : 21,
    'PcsOff' : 22,
    'IPR Fault Fast Frequency Change' : 23,
    'A_TestRN' : 24,
    'Wind Direction Transducer Error 1&3' : 25
}

def getList(dict): 
      
    return list(dict.keys()) 

alarm_description = getList(alarmdes_dict) #descriptions for alarms e.g. (DEMOTED) Yaw Brake 2 Under Pressure Full Brake (full alarm namers)

fevent_details_file = pd.read_csv("/data/faulteventdetails_latest.csv", header=0)

faultevent_details = list(fevent_details_file.Details) #descriptions for all fault events to replace <fevent-details> elements

faultevent_details

df_engcypher = pd.read_csv("/data/Original-Templates-BeforeParaphrasing.csv") #Read the original English-Cypher file (before preprocessing)
#mark rows that should that satisfy the conditions
df_engcypher["replace_subsysname"] = df_engcypher['Query'].str.contains("<subsys-name>") & df_engcypher['Code'].str.contains("<subsys-name>")

df_engcypher["replace_fngname"] = df_engcypher['Query'].str.contains("<fng-name>") & df_engcypher['Code'].str.contains("<fng-name>")


df_engcypher["replace_scadades"] = df_engcypher['Query'].str.contains("<scadadescription>") & df_engcypher['Code'].str.contains("<scadadescription>")

df_engcypher["replace_scadaname"] = df_engcypher['Query'].str.contains("<scadaname>") & df_engcypher['Code'].str.contains("<scadaname>")

df_engcypher["replace_scadafno"] = df_engcypher['Query'].str.contains("<scadafeatureno>") & df_engcypher['Code'].str.contains("<scadafeatureno>")

df_engcypher["replace_alarmno"] = df_engcypher['Query'].str.contains("<alarmno>") & df_engcypher['Code'].str.contains("<alarmno>")

df_engcypher["replace_alarmdes"] = df_engcypher['Query'].str.contains("<alarmdes>") & df_engcypher['Code'].str.contains("<alarmdes>")

df_engcypher["replace_feventdetails"] = df_engcypher['Query'].str.contains("<fevent-details>") & df_engcypher['Code'].str.contains("<fevent-details>")

pd.set_option('display.max_rows', 20) #Set maximum number of rows to be displayed (for checking whether replacements of tags is properly done)

dfs_list = []

for n in subsystem_names:
    aux = df_engcypher[df_engcypher["replace_subsysname"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<subsys-name>",n)
    aux["Code"] = aux["Code"].str.replace(r"<subsys-name>",n)
    dfs_list.append(aux)

for n in functionalgroup_names:
    aux = df_engcypher[df_engcypher["replace_fngname"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<fng-name>",n)
    aux["Code"] = aux["Code"].str.replace(r"<fng-name>",n)
    dfs_list.append(aux)

for n in scada_description:
    aux = df_engcypher[df_engcypher["replace_scadades"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<scadadescription>",n)
    aux["Code"] = aux["Code"].str.replace(r"<scadadescription>",n)
    dfs_list.append(aux)

for n in feature_labels:
    aux = df_engcypher[df_engcypher["replace_scadaname"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<scadaname>",n)
    aux["Code"] = aux["Code"].str.replace(r"<scadaname>",n)
    dfs_list.append(aux)

for n in feature_numbers:
    aux = df_engcypher[df_engcypher["replace_scadafno"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<scadafeatureno>",n)
    aux["Code"] = aux["Code"].str.replace(r"<scadafeatureno>",n)
    dfs_list.append(aux)

for n in alarm_numbers:
    aux = df_engcypher[df_engcypher["replace_alarmno"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<alarmno>",n)
    aux["Code"] = aux["Code"].str.replace(r"<alarmno>",n)
    dfs_list.append(aux)

for n in alarm_description:
    aux = df_engcypher[df_engcypher["replace_alarmdes"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<alarmdes>",n)
    aux["Code"] = aux["Code"].str.replace(r"<alarmdes>",n)
    dfs_list.append(aux)

for n in faultevent_details:
    aux = df_engcypher[df_engcypher["replace_feventdetails"]].copy()
    aux["Query"] = aux["Query"].str.replace(r"<fevent-details>",n)
    aux["Code"] = aux["Code"].str.replace(r"<fevent-details>",n)
    dfs_list.append(aux)

# Add the records which don't contain <> elements (fixed questions/queries) to the dataframe
dfs_list.append(df_engcypher[~((df_engcypher["replace_subsysname"])|(df_engcypher["replace_fngname"])|(df_engcypher["replace_scadades"])
|(df_engcypher["replace_scadaname"])|(df_engcypher["replace_scadafno"])|(df_engcypher["replace_alarmno"])
|(df_engcypher["replace_alarmdes"])|(df_engcypher["replace_feventdetails"]))])

replaced_df = pd.concat(dfs_list)

replaced_df.to_csv("/data/Queries-From-Templates.csv")

!pip install transformers==2.8.0

#Paraphrasing code is based on https://github.com/ramsrigouthamg/Paraphrase-any-question-with-T5-Text-To-Text-Transfer-Transformer- with some modifications
#Utilises a fine-tuned T5 base model (fine tuned on Quora Q/A dataset - so can be used for wind turbine questions paraphrasing)

import torch
print(torch.__version__)

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer


def set_seed(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

set_seed(42)

model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print ("device ",device)
model = model.to(device)

count_fcall = 0

def paraphraser_method(sentence):

  text =  "paraphrase: " + sentence + " </s>"


  max_len = 256

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=256,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=50 #Maximum number of sentences the model would return (paraphrased) - this is only max. (can be lower than this)
  )


  # print ("\nOriginal Question ::")
  # print (sentence)
  # print ("\n")
  # print ("Paraphrased Questions :: ")
  final_outputs =[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != sentence.lower() and sent not in final_outputs:
          final_outputs.append(sent)

  # for i, final_output in enumerate(final_outputs):
  #     print("{}: {}".format(i, final_output))
  global count_fcall
  count_fcall = count_fcall + 1
  print("Generating Paraphrases for",count_fcall," th dataframe sentence......")
  final_outputs.insert(0,sentence)
  return final_outputs

# replaced_df = replaced_df.iloc[:100,:]
replaced_df.Query.apply(str)
replaced_df.Code.apply(str)


replaced_df["Query"] = replaced_df["Query"].apply(paraphraser_method)

final_paraphrased_df = (replaced_df.set_index('Code')['Query'] # setting temporary index for exploding
 .explode().reset_index()
 .reindex(['Query', 'Code'], axis=1)) # ordering the columns

pd.set_option('display.max_rows', 100) #Set maximum number of rows to be displayed (for checking whether replacements of tags is properly done)

final_paraphrased_df

final_paraphrased_df.to_csv('/data/Final_Paraphrased_WindTurbine-Latest.csv')
