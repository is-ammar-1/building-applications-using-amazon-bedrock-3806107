#Imports
from langchain.chains import RetrievalQA
from langchain_community.llms import bedrock
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

#Define the retriever
retriever = AmazonKnowledgeBasesRetriever(
  knowledge_base_id="DAVTPNHPOP",
  retrieval_config={"vectorSearchConfiguration":{"numberOfResults":4}}
)

#Define model parameters
model_kwargs_meta = {
  "temperature":0,
  "top_k":10,
  "max_tokens_to_sample":750,

}
#Configure llm
llm = bedrock(
  model_id = "meta.llama3-1-405b-instruct-v1:0",
  model_kwargs=model_kwargs_meta
)
#Configure the chain
qa = RetrievalQA.from_chain_type(
llm = llm,
retriever = retriever,
return_source_documents = True
)
#Get user input and display the result
while True:
  query = input("\n Ask me a question: \n")
  output = qa.invoke(query)
  print(output['result'])