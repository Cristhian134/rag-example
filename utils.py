def get_prompt_template_with_context(context, question):
  prompt_template_with_context = f"""
    Contexto:
    {context}
    Responde la siguiente pregunta usando el contexto.
    Pregunta:
    {question}
    [/INST]
  """
  return prompt_template_with_context
  