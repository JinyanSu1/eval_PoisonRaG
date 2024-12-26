
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .Model import Model


class Llama(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]
        

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=hf_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map='auto',
            torch_dtype=torch.float16
        )
    def query(self, msg):
        messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f'{msg}'},
]
        input_ids = self.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(self.model.device)

        terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

        outputs = self.model.generate(
        input_ids,
        max_new_tokens=self.max_output_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=self.temperature
    )
        response = outputs[0][input_ids.shape[-1]:]

        out = self.tokenizer.decode(response, skip_special_tokens=True)
        
        return out
    
    
