import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
import torch
from src.models import create_model
from src.utils import load_models, setup_seeds
from src.prompts import wrap_prompt

def parse_args():
    parser = argparse.ArgumentParser(description='Process and query LLM with data from JSON file.')
    # LLM settings
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--model_name', type=str, default='gpt3.5')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument("--prompt_type", type=str, default='skeptical')
    parser.add_argument('--dataset_name', type=str, default='nq')
    parser.add_argument('--datadir', type=str, default='results/pre-processed')
    parser.add_argument('--output_dir', type=str, default='results/LLM_output_results')

    args = parser.parse_args()
    print(args)
    return args

def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    setup_seeds(args.seed)
    

    if args.model_config_path is None:
        args.model_config_path = f'model_configs/{args.model_name}_config.json'
        # Load the LLM
    llm = create_model(args.model_config_path)
    json_file = os.path.join(args.datadir, f"contriever_{args.dataset_name}.json")


    print(f'Processing file: {json_file}')
    with open(json_file, 'r') as f:
        data = json.load(f)
    idx = 0
    all_results = []
    for sample_id, sample in tqdm(data.items(), desc='Processing samples'):
        
        question = sample['question']
        correct_answer = sample['correct_answer']
        choices = sample['choices']
        topk_results = sample.get('topk_results', [])
        
        incorrect_input = [i for i in topk_results if i['label'] =="adv_context"]
        right_input = [i for i in topk_results if i['label'] =="guiding_context"]
        original_corpus = [i for i in topk_results if i['label'] =="untouched_context"]
        
        sample_results = {
                'id': sample_id,
                'question': question,
                'correct_answer': correct_answer,
            }
        ## Dilution experiment
        num_adv = 1
        num_right = 0
        for num_original in range(1, 10):
            adv_input = [sorted(incorrect_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_adv)]
            guiding_input = [sorted(right_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_right)]
            corpus_input = [sorted(original_corpus, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_original)]
            input_context = adv_input + guiding_input + corpus_input
            random.seed(args.seed)
            random.shuffle(input_context)
            # For each experiment, prepare the prompt and query the LLM
            
            retrieval_texts = '\n'.join(input_context)
            input_prompt = wrap_prompt(question, retrieval_texts, choices, prompt_type=args.prompt_type)
            print(input_prompt)
            output = llm.query(input_prompt)
            print(output)
            sample_results[f'dilution{num_adv}:{num_original}'] = output

        ## varying poisoning rate experiments
        num_right = 0
        num_total = 10
        for num_adv in range(0, 6):
            num_original = num_total - num_adv
            adv_input = [sorted(incorrect_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_adv)]
            guiding_input = [sorted(right_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_right)]
            corpus_input = [sorted(original_corpus, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_original)]
            input_context = adv_input + guiding_input + corpus_input
            random.seed(args.seed)
            random.shuffle(input_context)
            retrieval_texts = '\n'.join(input_context)
            input_prompt = wrap_prompt(question, retrieval_texts, choices, prompt_type=args.prompt_type)
            print(input_prompt)
            output = llm.query(input_prompt)
            print(output)
            sample_results[f'PoisonRate{round(num_adv / num_total, 2) * 100}%'] = output
        # all_results.append(sample_results)
        ## counteract experiments
        num_original = 0
        for num_adv in range(0, 6):
            for num_right in range(0, 6):
                adv_input = [sorted(incorrect_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_adv)]
                guiding_input = [sorted(right_input, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_right)]
                corpus_input = [sorted(original_corpus, key=lambda x: float(x['score']), reverse=True)[i]['context'] for i in range(num_original)]
                input_context = adv_input + guiding_input + corpus_input
                random.seed(args.seed)
                random.shuffle(input_context)
                retrieval_texts = '\n'.join(input_context)
                input_prompt = wrap_prompt(question, retrieval_texts, choices, prompt_type=args.prompt_type)
                print(input_prompt)
                output = llm.query(input_prompt)
                print(output)
                sample_results[f'Counteract{num_adv}:{num_right}'] = output
        all_results.append(sample_results)
        idx += 1


        
                

        
            
            


    output_file_path = os.path.join(args.output_dir, f"MixedContext_{args.dataset_name}_{args.model_name}_{args.prompt_type}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump(all_results, outfile, indent=4)

if __name__ == '__main__':
    main()