# https://developers.openai.com/cookbook/examples/search_reranking_with_cross-encoders?utm_source=chatgpt.com

import arxiv
from math import exp
import openai
import os
import pandas as pd
from tenacity import retry, wait_random_exponential, stop_after_attempt
import tiktoken
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

OPENAI_MODEL = "gpt-4o"

query = "how do bi-encoders work for sentence embeddings"
arxiv_client = arxiv.Client()
# Search describes the request; client.results(search) runs it and yields results.
search = arxiv.Search(
    query=query, max_results=20, sort_by=arxiv.SortCriterion.Relevance
)
result_list = []

for result in arxiv_client.results(search):
    result_dict = {}

    result_dict.update({"title": result.title})
    result_dict.update({"summary": result.summary})

    # Taking the first url provided
    result_dict.update({"article_url": [x.href for x in result.links][0]})
    result_dict.update({"pdf_url": [x.href for x in result.links][1]})
    result_list.append(result_dict)

for i, result in enumerate(result_list):
    print(f"{i + 1}: {result['title']}")

tokens = [" Yes", " No"]
tokenizer = tiktoken.encoding_for_model(OPENAI_MODEL)
ids = [tokenizer.encode(token) for token in tokens]
print(ids, ids[0], ids[1])


prompt = '''
You are an Assistant responsible for helping detect whether the retrieved document is relevant to the query. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved document is relevant to the query.

Query: How to plant a tree?
Document: """Cars were invented in 1886, when German inventor Carl Benz patented his Benz Patent-Motorwagen.[3][4][5] Cars became widely available during the 20th century. One of the first cars affordable by the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced horse-drawn carriages.[6] In Europe and other parts of the world, demand for automobiles did not increase until after World War II.[7] The car is considered an essential part of the developed economy."""
Relevant: No

Query: Has the coronavirus vaccine been approved?
Document: """The Pfizer-BioNTech COVID-19 vaccine was approved for emergency use in the United States on December 11, 2020."""
Relevant: Yes

Query: What is the capital of France?
Document: """Paris, France's capital, is a major European city and a global center for art, fashion, gastronomy and culture. Its 19th-century cityscape is crisscrossed by wide boulevards and the River Seine. Beyond such landmarks as the Eiffel Tower and the 12th-century, Gothic Notre-Dame cathedral, the city is known for its cafe culture and designer boutiques along the Rue du Faubourg Saint-Honoré."""
Relevant: Yes

Query: What are some papers to learn about PPO reinforcement learning?
Document: """Proximal Policy Optimization and its Dynamic Version for Sequence Generation: In sequence generation task, many works use policy gradient for model optimization to tackle the intractable backpropagation issue when maximizing the non-differentiable evaluation metrics or fooling the discriminator in adversarial learning. In this paper, we replace policy gradient with proximal policy optimization (PPO), which is a proved more efficient reinforcement learning algorithm, and propose a dynamic approach for PPO (PPO-dynamic). We demonstrate the efficacy of PPO and PPO-dynamic on conditional sequence generation tasks including synthetic experiment and chit-chat chatbot. The results show that PPO and PPO-dynamic can beat policy gradient by stability and performance."""
Relevant: Yes

Query: Explain sentence embeddings
Document: """Inside the bubble: exploring the environments of reionisation-era Lyman-α emitting galaxies with JADES and FRESCO: We present a study of the environments of 16 Lyman-α emitting galaxies (LAEs) in the reionisation era (5.8<z<8) identified by JWST/NIRSpec as part of the JWST Advanced Deep Extragalactic Survey (JADES). Unless situated in sufficiently (re)ionised regions, Lyman-α emission from these galaxies would be strongly absorbed by neutral gas in the intergalactic medium (IGM). We conservatively estimate sizes of the ionised regions required to reconcile the relatively low Lyman-α velocity offsets (ΔvLyα<300kms−1) with moderately high Lyman-α escape fractions (fesc,Lyα>5%) observed in our sample of LAEs, indicating the presence of ionised ``bubbles'' with physical sizes of the order of 0.1pMpc≲Rion≲1pMpc in a patchy reionisation scenario where the bubbles are embedded in a fully neutral IGM. Around half of the LAEs in our sample are found to coincide with large-scale galaxy overdensities seen in FRESCO at z∼5.8-5.9 and z∼7.3, suggesting Lyman-α transmission is strongly enhanced in such overdense regions, and underlining the importance of LAEs as tracers of the first large-scale ionised bubbles. Considering only spectroscopically confirmed galaxies, we find our sample of UV-faint LAEs (MUV≳−20mag) and their direct neighbours are generally not able to produce the required ionised regions based on the Lyman-α transmission properties, suggesting lower-luminosity sources likely play an important role in carving out these bubbles. These observations demonstrate the combined power of JWST multi-object and slitless spectroscopy in acquiring a unique view of the early stages of Cosmic Reionisation via the most distant LAEs."""
Relevant: No

Query: {query}
Document: """{document}"""
Relevant:
'''


# Build logit_bias from tokenizer so " Yes" / " No" IDs stay correct if the model changes.
logit_bias_yes_no = {ids[0][0]: 1, ids[1][0]: 1}


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def document_relevance(query, document):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt.format(query=query, document=document)}],
        temperature=0,
        logprobs=True,
        logit_bias=logit_bias_yes_no,
    )
    choice = response.choices[0]
    first_token_logprob = (
        choice.logprobs.content[0].logprob if choice.logprobs and choice.logprobs.content else None
    )
    return (
        query,
        document,
        choice.message.content,
        first_token_logprob,
    )

content = result_list[0]["title"] + ": " + result_list[0]["summary"]

# Set logprobs=True so the response includes token log-probabilities.
response = client.chat.completions.create(
    model=OPENAI_MODEL,
    messages=[{"role": "user", "content": prompt.format(query=query, document=content)}],
    temperature=0,
    logprobs=True,
    logit_bias=logit_bias_yes_no,
    max_tokens=1,
)

print('query:',query, '\ncontent', content)
result = response.choices[0]
print(f"Result was {result.message.content}")
if result.logprobs and result.logprobs.content:
    print(f"Logprobs was {result.logprobs.content[0].logprob}")
else:
    print("Logprobs was not returned")
print("\nBelow is the full logprobs object\n\n")
print(result.logprobs)

output_list = []
for x in result_list:
    content = x["title"] + ": " + x["summary"]

    try:
        output_list.append(document_relevance(query, document=content))

    except Exception as e:
        print(e)

# print(output_list[:5])

output_df = pd.DataFrame(
    output_list, columns=["query", "document", "prediction", "logprobs"]
).reset_index()
# Use exp() to convert logprobs into probability
output_df["probability"] = output_df["logprobs"].apply(exp)
# Reorder based on likelihood of being Yes
output_df["yes_probability"] = output_df.apply(
    lambda x: x["probability"] * -1 + 1
    if x["prediction"] == "No"
    else x["probability"],
    axis=1,
)
pd.set_option("display.max_columns", None)
print(output_df.head())