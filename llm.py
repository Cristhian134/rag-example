# load fine-tuned model from hub
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyLLM():

  tokenizer = None
  model = None

  def __init__(self):
    model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                device_map="auto",
                                                trust_remote_code=False,
                                                revision="main")

    config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
    model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

    self.model = model
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    self.model.eval()

  def get_response(self, prompt):
    inputs = self.get_inputs(prompt)
    outputs = self.model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
    response = self.tokenizer.batch_decode(outputs)[0]
    return response

  def get_tokenizer(self):
    return self.tokenizer

  def get_model(self):
    return self.model

  def get_inputs(self, prompt):
    inputs = self.tokenizer(prompt, return_tensors="pt")
    return inputs