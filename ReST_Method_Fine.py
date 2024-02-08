from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
import torch
import pandas as pd
import statistics
import random
import time

if torch.cuda.is_available():
    device = 0

def prepare_dataset(examples):
  input_ids = tokenizer(examples['input'], truncation=True, max_length=128)['input_ids']
  label_ids = tokenizer(examples["response"], truncation=True, max_length=128)['input_ids']
  return {"input_ids": input_ids, "labels": label_ids}

def dummy_reward_model(df):
    # Get input length
    input_len = [int(each.split()[0]) for each in df['input']]

    # Get word count of response
    gen_len = [len(each.split()) for each in df['response']]

    # Count number of "hugs" in response
    hug_count = [sum(1 for i in each.split() if i == 'hugs') for each in df['response']]
    
    # If count of "hugs" and total words in string do not match input legnth, give negative reward
    rewards = []
    for x,y,z in zip(input_len, hug_count, gen_len):
      if z == y:
        rewards.append(-abs(int(x) - y))
      else:
        rewards.append(-3)

    return rewards

def generate(df, gen_model, tokenizer, generation_kwargs, N):
    responses = []
    for _ in range(N):
        inputs = tokenizer(df['input'].tolist(), return_tensors="pt", padding = True, truncation = True)
        response = gen_model.generate(inputs["input_ids"].to(device), **generation_kwargs)
        response = [tokenizer.decode(each, skip_special_tokens = True) for each in response]
        responses.extend(response)

    # Create preserved-order dataframe with prompt / response pairs
    gen_df = pd.concat([df.copy()] * N)
    gen_df['response'] = responses

    return gen_df

def ReST(D, Deval, G, I, N, model, tokenizer, generation_kwargs, training_args):
    for g in range(G):
        print('Grow Step ', g)
        train_df = D.copy()

        train_df = generate(train_df, model, tokenizer, generation_kwargs, N)
        train_df['scores'] = dummy_reward_model(train_df)
        print(len(train_df[train_df['scores'] == 0]), "generations out of ", len(train_df), "are the correct length.")
        print("Example output from model:")
        print(train_df.head(50))
        time.sleep(10)

        steps = 0
        for tau_i in I:
            print('Improve Step: ', steps)
            print('Threshold: ', tau_i)
            
            # Filter for samples at or above threshold
            Dg = train_df.loc[(train_df['scores'] >= tau_i)].copy()
            if len(Dg) == 0:
                print("NO SAMPLES ABOVE THRESHOLD")
                break

            Dg = Dataset.from_pandas(Dg).map(prepare_dataset, batched=True)

            # Create trainer with newly filtered data
            trainer = Seq2SeqTrainer(model=model, args=training_args, tokenizer=tokenizer, train_dataset=Dg, data_collator=data_collator)

            # First fine-tuning of improve step
            trainer.train()

            # Evaluation
            Dg_e = Deval.copy()

            # Generate one response to for every sample in eval set
            Dg_e = generate(Dg_e, model, tokenizer, generation_kwargs, 1)
            Dg_e['scores'] = dummy_reward_model(Dg_e)
            
            prev = 0
            improve = statistics.mean(Dg_e['scores'])
            
            # While model improves reward model score on eval set, continue to fine-tune with newly filtered data
            while prev < improve:
                trainer.train()
                Dg_e = Deval.copy()
                Dg_e = generate(Dg_e, model, tokenizer, generation_kwargs, 1)
                Dg_e['scores'] = dummy_reward_model(Dg_e)
                prev = improve
                improve = statistics.mean(Dg_e['scores'])
            
            steps += 1
    
    print("Training Finished!!!")
    return model

def test_ReST(test_data, model, tokenizer, generation_kwargs):
      test_df = generate(test_data, model, tokenizer, generation_kwargs, 1)
      test_df['scores'] = dummy_reward_model(test_df)
      print("The fine-tuned model generates the correct number of hugs", len(test_df[test_df['scores'] == 0]) / len(test_df) * 100, "percent of the time!" )


rand_list = []
for i in range(1000):
     rand_list.append(str(random.randrange(1,5)) + ' hugs')

test_data = []
for i in range(100):
     test_data.append(str(random.randrange(1,5)) + ' hugs')

test_df = pd.DataFrame()
test_df['input'] = test_data

generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
}

training_args = Seq2SeqTrainingArguments(
            do_train=True,
            do_eval=False,
            learning_rate = 3e-4,
            output_dir="./t5-small",
            num_train_epochs=1,
            per_device_train_batch_size = 64
            )


tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, label_pad_token_id=-100)

train_df = pd.DataFrame()
train_df['input'] = rand_list
D, Deval = train_test_split(train_df, test_size = .1, random_state = 42)

G = 3
I = [-2,-1,0]
N = 10

fine_tuned_model = ReST(D, Deval, G, I, N, model, tokenizer, generation_kwargs, training_args)
test_ReST(test_df, fine_tuned_model, tokenizer, generation_kwargs)
