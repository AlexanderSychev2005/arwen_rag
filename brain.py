from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from config import LLM_PROVIDER, SYSTEM_INSTRUCTION
from memory import memory_store, retriever
from tools import (
    get_current_time,
    get_weather,
    search_the_web,
    send_telegram_message,
    tools_list,
)

print(f"Initializing {LLM_PROVIDER.upper()} Brain...")
# Initialize the model and bind our tools
chat_model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
llm_with_tools = chat_model.bind_tools(tools_list, parallel_tool_calls=False)


def get_llm_response(user_text: str) -> str:
    """
    Orchestrates the Agentic Loop. Rewrites the query, fetches context from
    lore and memory, and uses tools if necessary to formulate an answer.
    """
    try:
        # Step 1: Query Rewriting for precise RAG
        rewrite_prompt = f"""
        Extract the core search entities from this user message. 
        Convert it into a short, dense search query (2-5 words). 
        Do not answer the question, just output the keywords.
        User: '{user_text}'
        """
        optimized_query = chat_model.invoke(
            [HumanMessage(content=rewrite_prompt)]
        ).content.strip()

        # Step 2: Fetch Knowledge
        lore_docs = retriever.invoke(optimized_query)
        print(f"\n[RAG] Optimized Query: {optimized_query}")
        print(f"[RAG] Context found: {len(lore_docs)} paragraphs")

        lore_context: str = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}\nText: {doc.page_content}"
                for doc in lore_docs
            ]
        )

        memory_docs = memory_store.similarity_search(user_text, k=2)
        memory_context: str = "\n\n".join([doc.page_content for doc in memory_docs])

        # Step 3: Construct Prompt
        prompt: str = f"""
        PAST CONVERSATION MEMORIES:
        {memory_context if memory_docs else "No relevant memories found."}

        CONTEXT FROM MIDDLE-EARTH LORE (YOUR PRIMARY KNOWLEDGE):
        {lore_context if lore_docs else "No lore found in the archives."}

        User question: {user_text}
        """

        messages: list = [
            SystemMessage(content=SYSTEM_INSTRUCTION),
            HumanMessage(content=prompt),
        ]

        # Step 4: Agentic Loop
        max_iterations: int = 3
        current_iteration: int = 0

        while current_iteration < max_iterations:
            ai_msg = llm_with_tools.invoke(messages)

            if not ai_msg.tool_calls:
                return ai_msg.content.replace("*", "")

            print(f"[Arwen] Consulting the palantir (Step {current_iteration + 1})...")
            messages.append(ai_msg)

            for tool_call in ai_msg.tool_calls:
                tool_name: str = tool_call["name"]
                print(f"-> Arwen uses: {tool_name}")

                if tool_name == "get_weather":
                    location: str = tool_call["args"].get("location", "Kyiv")
                    tool_result: str = get_weather.invoke({"location": location})

                elif tool_name == "get_current_time":
                    tool_result: str = get_current_time.invoke({})

                elif tool_name == "search_the_web":
                    search_query: str = tool_call["args"].get("query", "")
                    print(f"-> Search query: {search_query}")
                    tool_result: str = search_the_web.invoke({"query": search_query})

                elif tool_name == "send_telegram_message":
                    tg_msg: str = tool_call["args"].get("message", "")
                    print(f"-> Sending to Telegram: {tg_msg}")
                    tool_result: str = send_telegram_message.invoke({"message": tg_msg})
                else:
                    tool_result = "Tool not found."

                messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
                )
            current_iteration += 1

        return "Sorry, the palantir shows too many confusing visions right now."

    except Exception as e:
        return f"Sorry, a shadow fell upon my mind: {e}"
