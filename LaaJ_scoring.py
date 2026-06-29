import re
import pandas as pd
from ollama import Client
import ast

client = Client(host='http://llm.ijs.si:11435')
eval_model = 'gemma3:4b-it-qat'

lang= 'Slovenian'
answers_path = r"C:\Users\mobrg\Documents\ProG\PycharmProjects\SloKuljko\candidate_answers_oneperrow.tsv"
candidate_ans = pd.read_csv(answers_path,sep='\t')

 # ["Fluency / Linguistic Acceptability","Faithfulness / Adequacy","Informativeness / Completeness", 'General Quality','"Notes"]
#metrike = ["Jezikovna ustreznost","Natančnost","Celovitost",'Splošna ocena',"Opombe"]
## USED once for processing unstructured criteria descriptions in prose
# criteria_path =r"C:\Users\mobrg\Documents\ProG\PycharmProjects\SloKuljko\LAAJ_criteria.txt"
# with open(criteria_path, 'r') as criteria:
#     textall = criteria.read()
#
# criteria_texts = textall.strip().split('\n\n')
# criteria = []
# for text in criteria_texts:
#     # Capture name and definition on the first line
#     header_match = re.match(r'^([\w\s/&]+?):\s*(.+)', text.strip())
#     if not header_match:
#         continue
#
#     criterion_name = header_match.group(1).strip()
#     metric_def = header_match.group(2).strip()
#
#     # Capture all score lines (handles multi-line score descriptions)
#     score_matches = re.findall(r'(\d):\s*(.+?)(?=\n\d:|$)', text, re.DOTALL)
#     scores = {int(k): v.strip() for k, v in score_matches}
#
#     criteria.append({
#         'name': criterion_name,
#         'definition': metric_def,
#         'scores': scores})
#
# # print result
# for m in criteria:
#     print(f"Criterion: {m['name']}")
#     print(f"Definition: {m['definition']}")
#     for score_val, score_desc in m['scores'].items():
#         print(f"  {score_val}: {score_desc}")
#     print()
# convert to tabular form, save to file
#criteria_df = pd.DataFrame.from_records(criteria)
#criteria_df.to_csv('LAAJ_criteria_tabular.tsv',sep='\t',encoding='UTF-8')

#read direclty from new file
criteria_df = pd.read_csv('LAAJ_criteria_tabular.tsv',sep='\t',encoding='UTF-8',index_col=0)

criteria = criteria_df.to_dict(orient='index')
#only_questions = questions['Question'].tolist()

#currently only for Slovenian (in English)
system_prompt = (f"You are fluent in {lang} and are an expert at evaluating answers about {lang} cultural knowledge."
                 f"Your evaluation is based on the provided reference answer and accounts for the context of {lang} language and culture.")

start_of_prompt = 'Evaluate the criterion of **{}** of the <hypothesis> answer to the <question> based on the provided <reference> answer.'
end_of_prompt = "\nProvide only the numerical score, without thinking or explaining. Score: "



def format_scores(scores):
    if isinstance(scores, str):
        scores = ast.literal_eval(scores)
    return '\n'.join(f"{k}: {v}" for k, v in scores.items())


def evaluate_row(row_df):
    question = row_df['Question']
    reference = row_df['CorrectAnswer']
    candidate = row_df['ModelAnswer']
    example = '<EXAMPLE><question>{}</question>\n<reference>{}</reference>\n<hypothesis>{}</hypothesis></EXAMPLE>\n'.format(question,reference,candidate)
    for i_crit in criteria.keys():
        crit = criteria[i_crit]

        prompt = (
                start_of_prompt.format(crit['name']) + '\n\n'
                                                       f"**Definition:** The **{crit['name']}** criterion {crit['definition']}\n\n"
                                                       f"**Score descriptions:**\n{format_scores(crit['scores'])}\n\n"
                + example
                + end_of_prompt
        )

        response = client.chat(
            model=eval_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        ).message.content

        row_df[crit['name']] = response.strip().strip('</score>')
    return row_df


candidate_ans_evald = candidate_ans.apply(evaluate_row,axis=1)

safe_model = eval_model.replace(':', '-')
candidate_ans_evald.to_csv(f'LAAJ_{safe_model}_{lang}.tsv', sep='\t')



