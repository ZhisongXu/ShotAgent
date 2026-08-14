import json

from llm import GPT4
from pipeline import prompts


def build_one_exp(degradations, experience):
    this_exp = f"To address {degradations} in the image, "
    for exe_path, stat in experience.items():
        plan = exe_path.split('+')
        degras, fail_rates = [], []
        for degra, fail_rate in stat['fail rate'].items():
            if degra != "total":
                degras.append(degra)
                fail_rates.append(f"{fail_rate:.0%}")
        this_exp += (
            f"when conducting first {plan[0]} and then {plan[1]}, "
            f"the fail rates of addressing {degras} are {fail_rates} respectively, "
            f"and the total fail rate is {stat['fail rate']['total']:.0%}; "
        )
    this_exp = this_exp[:-2] + '.'  # change "; " to "."
    return this_exp


with open("memory/fail_rate.json") as f:
    experience_hub = json.load(f)

exp_lst = []
for degras, exp in experience_hub.items():
    exp_lst.append(build_one_exp(degras, exp))
exp = '\n'.join(exp_lst)
prompt = prompts.distill_knowledge_prompt.format(experience=exp)
gpt = GPT4(system_message=prompts.system_message)
distilled = gpt(prompt=prompt)

schedule_experience = {
    "raw": exp,
    "distilled": distilled
}
print(prompt)
print()
print(distilled)

with open("memory/schedule_experience.json", "w") as f:
    json.dump(schedule_experience, f, indent=2)
