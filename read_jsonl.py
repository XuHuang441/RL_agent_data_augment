import json

file_path = "left10_outputs.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:  # 只看前三个
            break
        data = json.loads(line)
        print(json.dumps(data, indent=2, ensure_ascii=False))  # 美化输出
        llm_struct = json.loads(data["llm_struct"])

        # 现在就可以像正常字典一样访问
        print(llm_struct["analysis"])
        print(llm_struct["new_prompt"])
        print(llm_struct["reference_answer"])

