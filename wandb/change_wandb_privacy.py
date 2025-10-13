import os
import requests

api_key = os.environ["WANDB_API_KEY"]
query = """
mutation {
  updateProject(input: {entityName: "zeqi1213-brown-university", name: "rl_agent_gen", privacy: PUBLIC}) {
    project { name privacy }
  }
}
"""

r = requests.post(
    "https://api.wandb.ai/graphql",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={"query": query},
)
print(r.json())
