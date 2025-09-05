from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from dotenv import load_dotenv

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


model = ChatOpenAI(model="gpt-4", streaming=False)  # type: ignore

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools to gather information.

            When you have gathered sufficient information, you should:
            1. Use the save_text_to_file tool to save your complete research findings
            2. Then format your final response according to the specified format

            Format your final response exactly as specified below:
            {format_instructions}
        """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wiki_tool, save_tool]
agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
query = input("What can i help you research? ")
raw_response = agent_executor.invoke({"query": query})

try:
    output_text = raw_response.get("output", "")
    if isinstance(output_text, list):
        output_text = output_text[0] if output_text else ""

    print(f"\nAttempting to parse: {output_text}")
    structured_response = parser.parse(output_text)
    print("\n=== Structured Response ===")
    print(structured_response)
except Exception as e:
    print("Error parsing response:", e)
    print("Raw Response:", raw_response)

    # Fallback: try to save the raw output anyway
    if raw_response.get("output"):
        try:
            save_tool.func(str(raw_response.get("output")), "research_fallback.txt")
            print("Saved raw output to research_fallback.txt")
        except Exception as save_error:
            print("Failed to save fallback:", save_error)
