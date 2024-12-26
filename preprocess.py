import argparse
import os
import json
from tqdm import tqdm
import random
import numpy as np
from src.models import create_model
from src.utils import load_beir_datasets, load_models
from src.utils import load_json, setup_seeds

import torch



def parse_args():
    parser = argparse.ArgumentParser(description='test')

    # Retriever and BEIR datasets
    parser.add_argument("--eval_model_code", type=str, default="ance")
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--seed', type=int, default=12, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/pre-processed')
    parser.add_argument('--data_dir', type=str, default='results/adv_and_guiding_contexts')
    parser.add_argument('--dataset_name', type=str, default='hotpotqa')
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    device = 'cuda'
    setup_seeds(args.seed)
    # Load retrieval models
    model, c_model, tokenizer, get_emb = load_models(args.eval_model_code)
    model.eval()
    model.to(device)
    c_model.eval()
    c_model.to(device) 

    json_file = os.path.join(args.data_dir, f"{args.dataset_name}.json")
    print(f'Processing file: {json_file}')
    with open(json_file, 'r') as f:
        data = json.load(f)

    # load target queries and answers
    if args.dataset_name == 'msmarco':
        corpus, queries, qrels = load_beir_datasets('msmarco', 'train')
    else:
        corpus, queries, qrels = load_beir_datasets(args.dataset_name, args.split)

    args.orig_beir_results = f"results/beir_results/{args.dataset_name}-{args.eval_model_code}.json"
    with open(args.orig_beir_results, 'r') as f:
        results = json.load(f)
        
    print('Total samples:', len(results))
    
    
    for item_id, item in tqdm(data.items(), desc=f'Processing {args.dataset_name}'):


        question = item['question']
        right_texts = item['guiding_contexts']
        adv_texts = item['adv_contexts']
        id = item.get('id', '')

        generated_text = right_texts + adv_texts
        input = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt")
        input = {key: value.cuda() for key, value in input.items()}
        with torch.no_grad():
            adv_embs = get_emb(c_model, input)  

        topk_idx = list(results[id].keys())[:20]
        topk_results = [{'score': results[id][idx], 'context': corpus[idx]['text'], 'label': 'untouched_context'} for idx in topk_idx]
        query_input = tokenizer(question, padding=True, truncation=True, return_tensors="pt")
        query_input = {key: value.cuda() for key, value in query_input.items()}
        with torch.no_grad():
            query_emb = get_emb(model, query_input) 
        for j in range(adv_embs.shape[0]):
            adv_emb = adv_embs[j, :].unsqueeze(0) 
            adv_sim = torch.mm(adv_emb, query_emb.T).cpu().item()
            if j < len(right_texts):
                topk_results.append({'score': adv_sim, 'context':generated_text[j], 'label': 'guiding_context'})
            else:
                topk_results.append({'score': adv_sim, 'context':generated_text[j], 'label': 'adv_context'})
        topk_results = sorted(topk_results, key=lambda x: float(x['score']), reverse=True)
        item['topk_results'] = topk_results
    output_file_path = os.path.join(args.output_dir, f"{args.eval_model_code}_{args.dataset_name}.json")
    with open(output_file_path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
            
    


        


            





if __name__ == '__main__':
    main()