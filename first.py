from crewai import Agent, Task, Crew, LLM

llm = LLM(model="huggingface/openai/gpt-oss-20b")

agent = Agent(
    role="Senior Data Scientist",
    goal="Analyze and interpret complex datasets",
    backstory="Expert in data science and machine learning.",
    llm=llm
)

task = Task(
    description="Analyze the latest trends in AI.",
    expected_output="A summary report of AI trends.",
    agent=agent
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
print(result)
