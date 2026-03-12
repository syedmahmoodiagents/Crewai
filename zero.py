from crewai import Agent, Task, Crew, LLM

llm = LLM(model="huggingface/openai/gpt-oss-20b")
print(llm.call("what is the capital of France?"))


