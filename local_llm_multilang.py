import os
import pandas as pd
from ollama import Client
import json

with open("lang_prompts.json", "r", encoding="utf-8") as f:
    SYSTEM_PROMPTS = json.load(f)

client = Client(host='http://llm.ijs.si:11435')

MODELS = ["gemma3:27b-it-qat","llama3.3:latest"]
common_cols = ['Category','Question','CorrectAnswer']

sample_dir = '.data/samples'


for filename in os.listdir(sample_dir):
    if filename.endswith(".tsv"):
        lang = filename.split('_')[1]
        #if lang == "Slovenian":
        questions = pd.read_csv(os.path.join(sample_dir,filename),sep='\t')

        only_questions = questions['Question'].tolist()
        lang_cfg = SYSTEM_PROMPTS[lang]
        system_prompt = lang_cfg["system"]
        question_label = lang_cfg["question_label"]

        for model in MODELS:
            answers = []
            for question in only_questions:
                full_question = (f"{question_label}:\n"
                                 f"{question}\n"
                                 f": ")
                response = client.chat(model=model,
                                       messages=[{'role': "system", 'content': system_prompt},
                                                 {'role': "user", 'content': full_question}]).message.content
                answers.append(response)
            questions[model] = answers


        questions.to_csv(f'.data/model_ans/candidate_answers_sample_{lang}.txt',sep='\t',index=False)


        # for fajl in os.listdir('.data/model_ans'):
        #     lang = fajl.split('_')[-1].strip('.txt')
        #     if lang == "Flemish":
        #         questions = pd.read_csv(os.path.join('.data/model_ans',fajl),sep='\t')
        annotation_df = questions.melt(
            id_vars=["Question", "CorrectAnswer"],
            value_vars=MODELS,
            var_name="Model",
            value_name="ModelAnswer"
        )

        annotation_df["Fluency"] = ""
        annotation_df["Adequacy"] = ""
        annotation_df["General_Quality"] = ""
        annotation_df["Comments"] = ""
        annotation_df.to_excel(f".data/for_eval/for_manual_evaluation_{lang}.xlsx", index=False)