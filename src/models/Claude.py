from openai import OpenAI
from .Model import Model
import anthropic



class Claude(Model):
    def __init__(self, config):
        super().__init__(config)
        api_keys = config["api_key_info"]["api_keys"]
        api_pos = int(config["api_key_info"]["api_key_use"])
        assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.client = anthropic.Anthropic(api_key=api_keys[api_pos])

    def query(self, msg):
        try:
            message = self.client.messages.create(
                    model=self.name,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": msg}
                    ]
                )

# Print the response
            response = message.content[0].text
            
           
        except Exception as e:
            print(e)
            response = ""

        return response