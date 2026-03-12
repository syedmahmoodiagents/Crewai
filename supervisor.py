import os
from dotenv import load_dotenv
load_dotenv()  
os.getenv("HF_TOKEN")

from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

llm = LLM(model="huggingface/meta-llama/Meta-Llama-3-8B-Instruct")

@tool
def addition(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@tool
def subtraction(a: int, b: int) -> int:
    """Subtract two numbers"""
    return a - b

worker_agent = Agent(
    role="Math Worker",
    goal="Perform mathematical operations using available tools",
    backstory="You are good at calculations using tools.",
    tools=[addition, subtraction],
    llm=llm,
    verbose=True
)

supervisor_agent = Agent(
    role="Supervisor",
    goal="Understand the user question and assign the correct calculation task",
    backstory="You are responsible for deciding what needs to be calculated.",
    llm=llm,
    verbose=True,
    allow_delegation=True,
    function_calling_llm=llm
)

task1 = Task(
    description="Calculate 10 + 5",
    agent=worker_agent,
    expected_output="Result of addition"
)

task2 = Task(
    description="Calculate 20 - 7",
    agent=worker_agent,
    expected_output="Result of subtraction"
)

task = Task(
    description="Now add the following: (10 + 5) and (20 - 7)",
    agent=supervisor_agent,
    expected_output="Final calculations"
)

crew = Crew(
    agents=[supervisor_agent, worker_agent],
    tasks=[task1, task2, task],
    manager_agent=supervisor_agent,
    verbose=True
)

result = crew.kickoff()
print(result)