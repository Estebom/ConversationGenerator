
import os
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chains import LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import StringPromptTemplate
from langchain.utilities import SerpAPIWrapper    
from typing import List
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

search = SerpAPIWrapper()
@tool
def update_conversation(tool_input):
    """
    Updates the state of the conversation with new responses.

    Args:
        intermediate_steps: The current state of the conversation.
        new_user_1_response: The latest response from User 1.
        new_user_2_response: The latest response from User 2.

    Returns:
        The updated conversation state.
    """
    intermediate_steps = tool_input["intermediate_steps"]
    new_user_1_response = tool_input["User_1_response"]
    new_user_2_response = tool_input["User_2_response"]
    new_step = (new_user_1_response, new_user_2_response)
    intermediate_steps.append(new_step)
    return intermediate_steps

tools = [
    Tool(
        name = "Search",
        func = search.run,
        description= "useful for when you need to present a conversation topic on current events"

    ),
    Tool(
        name = "UpdateConversation",
        func = update_conversation,
        description= "updates the state of the conversation with new responses"

    )
]


template ="""\
Given the two user's data that includes interests and traits, find a current discussion topic and create a 16 sentense discussion between the two users using realistic emotional language. These are the tools available to you:
{tools}



Use the following format:

Topic: the current discussion topic based on user_1 and user_2 
Thought: You should always think about what to do 
Action: the action to take should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I know what the conversation will be about
Conversation Start: imitate User_1 to start discussion based on observation 
Thought: I have User_1's conversation starter
User_2 Response: imitate User_2 reply to User_1's Conversation starter
Thought: I have User_2's reply to User_1's response
User_1 Response: imitate User_1 reply to User_1's reply
Thought: I have User_1's reply to User_2's reponse
... (this Thought/User_2 Response/ User_1 Response should repeat 8 times)
...(have the two users reply to each other until there are 16 lines)
Thought: I now have the conversation 
Final Conversation: the final conversation based on user_1 and user_2

Begin! Remember to have the personalities of each user in mind when giving your final conversation.


User_1 = {user_1}
User_2 = {user_2}
{agent_scratchpad}"""


from typing import Callable


class ConvoPromptTemplate(StringPromptTemplate):
    template: str

    tools:List[Tool]

    def format(self, **kwargs) -> str:
        user_1 = kwargs.get("user_1", "")
        user_2 = kwargs.get("user_2","")
        intermediate_steps = kwargs.get("intermediate_steps",[])

        prompt = self.template.format(

            user_1 = user_1,
            user_2 = user_2,
            tools = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
            tool_names = ", ".join([tool.name for tool in self.tools]),
            agent_scratchpad = self.generate_conversation(intermediate_steps)
        )
        return prompt
    
    def generate_conversation(self, intermediate_steps):
        conversation = ""
        #ADD THOUGHT HERE
        for step in intermediate_steps:
            # Unpack the step tuple
            ##ESTEBAN JUST MAKE SURE TO UNPACK ACTION AND OBSERVATION
            if len(step) == 2:
                action, observation = step
                conversation += f"\nAction: {action}\nThought:"
                conversation += f"\nObservation: {observation}:\nThought:"
            elif len(step) == 5:
                action, observation, conversation_start, User_2, User_1 = step
                # conversation += f"\nAction: {action}\nThought:"
                # conversation += f"\nObservation: {observation}:\nThought:"
                conversation += f"\nConversation Start: {conversation_start}\nThought:"
                conversation += f"\nUser_2 Response: {User_2}\nThought:"
                conversation += f"\nUser_1 Response: {User_1}\nThought:"

            # Check if the conversation length is less than 16 lines
            
        if len(conversation) < 8:
            conversation += "\n... (continue the conversation)"

        return conversation
      
         
prompt = ConvoPromptTemplate(
    template=template,
    tools = tools,
    input_variables=["user_1", "user_2","intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def extract_response_for_user_1(self,llm_output):
        """
        Extracts User_1's response from the language model's output.

        Args:
            llm_output (str): the raw output from the language model.

        Returns:
            str: The extracted response of User_1.
        """
        pattern = r"User_1 Response: (.*?)\n"

        # Use regular expression to search for the pattern
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            # Return the captured group which is User_1's response
            return match.group(1).strip()
        else:
            # Handle cases where the pattern is not found
            return "No response found"  # or handle it differently as needed
        
    def extract_response_for_user_2(self,llm_output):
        """
        Extracts User_2's response from the language model's output.

        Args:
            llm_output (str): the raw output from the language model.

        Returns:
            str: The extracted response of User_2.
        """
        pattern = r"User_2 Response: (.*?)\n"

        # Use regular expression to search for the pattern
        match = re.search(pattern, llm_output, re.DOTALL)
        if match:
            # Return the captured group which is User_1's response
            return match.group(1).strip()
        else:
            # Handle cases where the pattern is not found
            return "No response found"  # or handle it differently as needed

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish with Observation
        if "Observation:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Observation:")[-1].strip()},
                log=llm_output,
            )

        # Check for User Responses
        elif "User_1 Response" in llm_output or "User_2 Response" in llm_output:
            new_user_1_response = self.extract_response_for_user_1(llm_output)
            new_user_2_response = self.extract_response_for_user_2(llm_output)
            return AgentAction(
                tool="UpdateConversation",
                tool_input={"intermediate_steps": intermediate_steps,"User_1_response": new_user_1_response, "User_2_response": new_user_2_response},
                log=llm_output
            )

        # Check if agent should finish with Final Conversation
        elif "Final Conversation:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Conversation:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if match:
            action = match.group(1).strip()
            action_input = match.group(2)
            return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)
        else:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
    


outputParser = CustomOutputParser()

llm = OpenAI(temperature = 0.9)

llm_chain = LLMChain(llm = llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
llm_chain=llm_chain,
output_parser=outputParser,
stop=["\nObservation:"],
allowed_tools = tool_names

)

from langchain.schema.agent import AgentFinish
from langchain.agents import AgentExecutor

# docs = [
#     Document(page_content=t.description, metadata={"index": i})
#     for i, t in enumerate(tools)
# ]
# vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
# retriever = vector_store.as_retriever()


# def get_tools(query):
#     docs = retriever.get_relevant_documents(query)
#     return [tools[d.metadata["index"]] for d in docs]

intermediate_steps = []
user_1 =  "Office worker that loves to polay video games, on his days off he enjoys watching anime"
user_2 = "Teacher that loves to do art in free time. Always up to date on politics"
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools,handle_parsing_errors=True, verbose = True)
conversation = agent_executor.run(prompt = prompt, user_1= user_1, user_2 = user_2)
print(conversation)
# while True:
#     current_prompt = prompt.format(user_1 = user_1, user_2 = user_2, intermediate_steps = intermediate_steps)

#     response,get_tools =  agent_executor.run(prompt = current_prompt, user_1 = user_1, user_2 = user_2)

#     if get_tools == "UpdateConversation":

        

#         intermediate_steps = update_conversation(
#             intermediate_steps,
#             ["User_1_response"],
#             ["User_2_response"]
            
#         )

#         if len(intermediate_steps) >=8:
#             break
# final_conversation = final_conversation = "\n".join([f"{step[0]} {step[1]}" for step in intermediate_steps])
# print("Final Conversation:" , final_conversation)




