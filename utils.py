from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.llms import HuggingFaceHub
from langchain.agents.agent_types import AgentType

def query_agent(data, query):
    # Parse the CSV file and create a Pandas DataFrame from its contents.
    df = pd.read_csv(data)

    # Set the desired parameters for the HuggingFaceHub model
    model_kwargs = {
        'temperature': 0.2
    }

    # Create a HuggingFaceHub instance with the specified parameters
    llm = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-alpha", model_kwargs=model_kwargs)

    # Create a Pandas DataFrame agent.
    agent = create_pandas_dataframe_agent(llm, 
                                          df, 
                                          verbose=True, 
                                          handle_parsing_errors=True
                                        )

    # Python REPL: A Python shell used to evaluating and executing Python commands.
    # It takes python code as input and outputs the result. The input python code can be generated from another tool in the LangChain
    
    query = query + " using tool python_repl_ast and stop generation when you generate the final answer"
    
    # Run the agent with the modified HuggingFaceHub instance
    return agent.run(query)