# https://developers.openai.com/cookbook/examples/responses_api/responses_api_tool_orchestration
import os
import time
from tqdm.auto import tqdm
import pandas as pd
from pandas import DataFrame
from datasets import load_dataset
import random
import string
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
# Import Pinecone client and related specifications.
from pinecone import Pinecone
from pinecone import ServerlessSpec

# Load .env from Random/ so OPENAI_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load the dataset (set HF_TOKEN in .env for higher rate limits and faster downloads)
ds = load_dataset(
    "FreedomIntelligence/medical-o1-reasoning-SFT",
    "en",
    split="train[:100]",
    token=os.getenv("HF_TOKEN"),
)
ds_dataframe = DataFrame(ds)
# print(ds_dataframe)
# Merge the Question and Response columns into a single string.
ds_dataframe['merged'] = ds_dataframe.apply(
    lambda row: f"Question: {row['Question']} Answer: {row['Response']}", axis=1
)

print("Example merged text:", ds_dataframe['merged'].iloc[0])

MODEL = "text-embedding-3-small"  # Replace with your production embedding model if needed
# Compute an embedding for the first document to obtain the embedding dimension.
sample_embedding_resp = client.embeddings.create(
    input=[ds_dataframe['merged'].iloc[0]],
    model=MODEL
)
embed_dim = len(sample_embedding_resp.data[0].embedding)
print(f"Embedding dimension: {embed_dim}")


##################################################################
# Initialize Pinecone using your API key.
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# Define the Pinecone serverless specification.
AWS_REGION = "us-east-1"
spec = ServerlessSpec(cloud="aws", region=AWS_REGION)

# Create a random index name with lower case alphanumeric characters and '-'
index_name = 'pinecone-index-multi-tool'# + ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))

# Create the index if it doesn't already exist.
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=embed_dim,
        metric='dotproduct',
        spec=spec
    )

# Connect to the index.
index = pc.Index(index_name)
time.sleep(1)
print("Index stats:", index.describe_index_stats())


batch_size = 32
for i in tqdm(range(0, len(ds_dataframe['merged']), batch_size), desc="Upserting to Pinecone"):
    i_end = min(i + batch_size, len(ds_dataframe['merged']))
    lines_batch = ds_dataframe['merged'][i: i_end]
    ids_batch = [str(n) for n in range(i, i_end)]
    
    # Create embeddings for the current batch.
    res = client.embeddings.create(input=[line for line in lines_batch], model=MODEL)
    embeds = [record.embedding for record in res.data]
    
    # Prepare metadata by extracting original Question and Answer.
    meta = []
    for record in ds_dataframe.iloc[i:i_end].to_dict('records'):
        q_text = record['Question']
        a_text = record['Response']
        # Optionally update metadata for specific entries.
        meta.append({"Question": q_text, "Answer": a_text})
    
    # Upsert the batch into Pinecone.
    vectors = list(zip(ids_batch, embeds, meta))
    index.upsert(vectors=vectors)


def query_pinecone_index(client, index, model, query_text):
    # Generate an embedding for the query.
    query_embedding = client.embeddings.create(input=query_text, model=model).data[0].embedding

    # Query the index and return top 5 matches.
    res = index.query(vector=[query_embedding], top_k=5, include_metadata=True)
    print("Query Results:")
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata'].get('Question', 'N/A')} - {match['metadata'].get('Answer', 'N/A')}")
    return res

############## Example usage with a different query from the train/test set
# query = (
#     "A 45-year-old man with a history of alcohol use presents with symptoms including confusion, ataxia, and ophthalmoplegia. "
#     "What is the most likely diagnosis and the recommended treatment?"
# )
# query_pinecone_index(client, index, MODEL, query)

# # Retrieve and concatenate top 3 match contexts.
# matches = index.query(
#     vector=[client.embeddings.create(input=query, model=MODEL).data[0].embedding],
#     top_k=3,
#     include_metadata=True
# )['matches']

# context = "\n\n".join(
#     f"Question: {m['metadata'].get('Question', '')}\nAnswer: {m['metadata'].get('Answer', '')}"
#     for m in matches
# )

# Use the context to generate a final answer.
# response = client.responses.create(
#     model="gpt-4o",
#     input=f"Provide the answer based on the context: {context} and the question: {query} as per the internal knowledge base",
# )

# print("\nFinal Answer:")
# print(response.output_text)

# Tools definition: The list of tools includes:
# - A web search preview tool.
# - A Pinecone search tool for retrieving medical documents.

# Define available tools.
tools = [   
    {"type": "web_search",
      "user_location": {
        "type": "approximate",
        "country": "US",
        "region": "California",
        "city": "SF"
      },
      "search_context_size": "medium"},
    {
        "type": "function",
        "name": "PineconeSearchDocuments",
        "description": "Search for relevant documents based on the medical question asked by the user that is stored within the vector database using a semantic query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The natural language query to search the vector database."
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return.",
                    "default": 3
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
]

# Example queries that the model should route appropriately.
queries = [
    {"query": "which team won the latest superbowl?"},
    {"query": "What is the most common cause of death in the United States according to the internet?"},
    {"query": ("A 7-year-old boy with sickle cell disease is experiencing knee and hip pain, "
               "has been admitted for pain crises in the past, and now walks with a limp. "
               "His exam shows a normal, cool hip with decreased range of motion and pain with ambulation. "
               "What is the most appropriate next step in management according to the internal knowledge base?")}
]


# Collect tool-call stats from each response (tool name, type, query/input).
tool_calls = []

# Process each query dynamically.
for item in queries:
    input_messages = [{"role": "user", "content": item["query"]}]
    print("\n🌟--- Processing Query ---🌟")
    print(f"🔍 **User Query:** {item['query']}")
    
    # Call the Responses API with tools enabled and allow parallel tool calls.
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "When prompted with a question, select the right tool to use based on the question."
            },
            {"role": "user", "content": item["query"]}
        ],
        tools=tools,
        parallel_tool_calls=True
    )
    
    print("\n✨ **Initial Response Output:**")
    print(response.output)

    # Collect tool-call stats from this response (only web_search_call and function_call).
    for i in response.output:
        item_type = getattr(i, "type", None)
        if item_type not in ("web_search_call", "function_call"):
            continue
        call_id = getattr(i, "call_id", None) or getattr(i, "id", None)
        if item_type == "web_search_call":
            tool_name = "web_search"
            query_input = "N/A"
            if hasattr(i, "action") and i.action is not None:
                query_input = getattr(i.action, "queries", None) or getattr(i.action, "query", "N/A")
                if isinstance(query_input, list):
                    query_input = query_input[0] if query_input else "N/A"
        else:
            tool_name = getattr(i, "name", "N/A")
            query_input = getattr(i, "arguments", "N/A")
        tool_calls.append({
            "Type": item_type,
            "Call ID": call_id,
            "Tool": tool_name,
            "Query/Input": str(query_input)[:200] + ("..." if len(str(query_input)) > 200 else ""),
        })

    # Determine if a tool call is needed and process accordingly.
    if response.output:
        tool_call = response.output[0]
        if tool_call.type in ["web_search_call", "function_call"]:
            tool_name = tool_call.name if tool_call.type == "function_call" else "web_search_call"
            print(f"\n🔧 **Model triggered a tool call:** {tool_name}")
            
            if tool_name == "PineconeSearchDocuments":
                print("🔍 **Invoking PineconeSearchDocuments tool...**")
                res = query_pinecone_index(client, index, MODEL, item["query"])
                if res["matches"]:
                    best_match = res["matches"][0]["metadata"]
                    result = f"**Question:** {best_match.get('Question', 'N/A')}\n**Answer:** {best_match.get('Answer', 'N/A')}"
                else:
                    result = "**No matching documents found in the index.**"
                print("✅ **PineconeSearchDocuments tool invoked successfully.**")
                # Append the function call and its output, then get final answer from the model.
                input_messages.append(tool_call)
                input_messages.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": str(result)
                })
                final_response = client.responses.create(
                    model="gpt-4o",
                    input=input_messages,
                    tools=tools,
                    parallel_tool_calls=True
                )
                print("\n💡 **Final Answer:**")
                print(final_response.output_text)
            else:
                # web_search_call: the API already ran the search and returned the answer in this response.
                # Do not send function_call_output; the API does not expect it for web search.
                print("🔍 **Web search was used by the API.**")
                print("\n💡 **Final Answer:**")
                print(response.output_text)
        else:
            # If no tool call is triggered, print the response directly.
            print("💡 **Final Answer:**")
            print(response.output_text)



# Display tool-call stats across all processed queries.
print("\n" + "=" * 60)
print("📊 Tool usage stats (all queries)")
print("=" * 60)
if tool_calls:
    df_tool_calls = pd.DataFrame(tool_calls)
    print(df_tool_calls.to_string(index=False))
else:
    print("No tool calls were used in the processed queries.")

