import json
import os
from openai import OpenAI
from utils import format_input
from tqdm import tqdm
import time

def run_gpt4o_mini(prompt, client, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=123,
    )
    return response.choices[0].message.content

def generate_scores(json_data, client):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the number only."
        )
        score = run_gpt4o_mini(prompt, client)
        try:
            scores.append(int(score))
            print(f"Average score: {sum(scores) / len(scores)}\n")
        except ValueError:
            continue
        time.sleep(25)

    return scores


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    #prompt = f"Respond with 'hello world' if you got this message."
    #print(run_gpt4o_mini(prompt, client))

    json_file = "model_responses/model_responses_1.json"

    with open(json_file, "r") as file:
        json_data = json.load(file) 

    scores = generate_scores(json_data, client)
    print(f"Average score: {sum(scores) / len(scores):.3f}")
    with open("model_1_scores.json", "w") as f:
        json.dump(scores, f)






