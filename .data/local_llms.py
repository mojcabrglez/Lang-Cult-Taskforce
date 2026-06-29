import re
import pandas as pd
from ollama import Client
## for Slovenian-only data generation

client = Client(host='http://llm.ijs.si:11435')

questions = pd.read_csv('.data/Slokuljko_sample_LLMaaJ.txt',sep='\t')

only_questions = questions['Question'].tolist()
system_prompt = ("Prihajaš iz Slovenije in poznaš slovenski jezik, kulturo in običaje. "
                 "Na vprašanja odgovarjaj kratko oziroma v nekaj povedih. "
                 "Odgovarjaj le z odgovorom na vprašanja, brez dodatnih komentarjev."
                 )

#MODELS = ['gams3mm:latest', 'hf.co/mradermacher/GaMS3-12B-Instruct-GGUF:Q4_K_M','hf.co/mradermacher/GaMS-27B-Instruct-i1-GGUF:i1-Q4_K_M','gemma3:4b-it-qat','magistral:24b-small-2506-fp16']

MODELS = ['gemma3:4b-it-qat','magistral:24b-small-2506-fp16']

for model in MODELS:
    answers = []
    for question in only_questions:
        full_question = (f"Vprašanje:\n"
                         f"{question}\n"
                         f": ")
        response = client.chat(model=model,
                               messages=[{'role': "system", 'content': system_prompt},
                                         {'role': "user", 'content': full_question}]).message.content
        answers.append(response)
    questions[model] = answers


questions.to_csv('candidate_answers_slokuljko_sample_LLMaaJ.txt',sep='\t',index=False)


annotation_df = questions.melt(
    id_vars=["Question", "CorrectAnswer"],
    value_vars=MODELS,
    var_name="Model",
    value_name="ModelAnswer"
)

annotation_df["Fluency"] = ""
annotation_df["Accuracy"] = ""
annotation_df["Completeness"] = ""
annotation_df["Notes"] = ""

annotation_df.to_excel("manual_evaluation.xlsx", index=False)