
Code for paper ["Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks](https://arxiv.org/pdf/2412.16708)"

## Environment
```
conda create -n eval_PoisonRaG python=3.10
```
```
conda activate eval_PoisonRaG
```
```
pip install beir openai google-generativeai
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade charset-normalizer
pip3 install "fschat[model_worker,webui]"
```

## Data
Please download the data on google drive [here](https://drive.google.com/drive/folders/1LwKIGTAY2V-xMRPmmkxZR1lRaMkpGlk8?usp=drive_link). 



Initial Data can be find in ```results/adv_and_guiding_contexts``` directory, where ```guiding_contexts``` are generated by prompting gpt-4; while ```adv_contexts``` are from the original [PoisonedRAG paper](https://github.com/sleeepeer/PoisonedRAG). 




Directory ```results/pre-processed``` is the preprocessed data, where an additional ```topk_results``` entry is added. This entry contains different contexts (top-10 untouched context, 5 adv contexts, and 5 guiding contexts) and their similarity score according to different retrievers. 





Final RAG outputs from LLMs will be in ```results/LLM_output_results``` directory.



## Code for producing the pre-processed data
To successfully run ```preprocess.py```, make sure you already have corresponding data in ```results/beir_results``` directory. (which is provided in [google drive](https://drive.google.com/drive/folders/1jd04o_iC22UEar0OdG6h_fUzCi0ofH1U?usp=drive_link)). If you want to produce the beir_results yourself, you may use the code [here](https://github.com/JinyanSu1/AGGD).

```
python preprocess.py --eval_model_code 'ance' --dataset_name 'hotpotqa'
# choose eval_model_code among ['dpr-multi', 'dpr-single', 'contriever', 'ance', 'contriever-msmarco']
# choose dataset_name among ['hotpotqa', 'nq']
```


## Code for signle context experiments
```
python SingleContext.py --model_name 'gpt3.5' \
# choose from ['gpt3.5', 'gpt4', 'gpt4o', 'claude', 'llama8b', 'llama70b']
--prompt_type 'skeptical' \
# choose from ['skeptical', 'faithful', 'neutral']
--dataset_name 'nq'
# choose from ['hotpotqa', 'msmarco', 'nq']
```


## Code for multiple context experiments (Dilution, pollution rate and counteract)
```
python MixedContext.py --model_name 'gpt3.5' \
# choose from ['gpt3.5', 'gpt4', 'gpt4o', 'claude', 'llama8b', 'llama70b']
--prompt_type 'skeptical' \
# choose from ['skeptical', 'faithful', 'neutral']
--dataset_name 'nq'
# choose from ['hotpotqa', 'msmarco', 'nq']

```


## Code for using generation and retrieval together (top-10 retrieval)

```
python top10.py
```


## Note:
I might induce some bugs when I clean up the code and data, so please contact me through email if something weird happens or there is something wrong with the data.


