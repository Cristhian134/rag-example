from retriever import MyRetriever
from llm import MyLLM
from utils import get_prompt_template_with_context

class MyRAG:
  def __init__(self):
    self.retriever = MyRetriever()
    self.llm = MyLLM()
  
  def get_answer(self, question):
    prompt = self.get_prompt(question)
    return self.llm.get_response(prompt)

  def get_prompt(self, question):
    context = self.retriever.get_context(question)
    # prompt (with context)
    prompt = get_prompt_template_with_context(context, question)
    return prompt