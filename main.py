from rag import MyRAG

rag = MyRAG()

while True:
  question = input("Question: ")
  if question == "0":
    break
  answer = rag.get_answer(question)
  print(answer)

# question = "What is fat-tailedness?"
# rag.get_answer(question)