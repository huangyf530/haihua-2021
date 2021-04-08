# -*- coding: UTF-8 -*- 
import json
import os
import csv

def format_article(article_json):
    content = article_json['Content']
    question_ls = []
    for i in article_json['Questions']:
        query = i['Question']

        ans = None
        if 'Answer' in i:
            ans = ord(i['Answer']) - ord('A')
        quesiton_id =i['Q_id']

        for j, option in enumerate(i['Choices']):
            score = 1 if ans == j else 0
            format_str = f"[CLS] {content} [SEP] {query} {option} [SEP]"

            if ans is not None:
                question_ls.append((quesiton_id, format_str, score))
            else:
                question_ls.append((quesiton_id, format_str))

    return question_ls

if __name__ == '__main__':
    train_path = "data/train.json"
    train_csv_path = "data/train.csv"

    validation_path = "data/validation.json"
    validation_csv_path = "data/validation.csv"

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'score'])

        for i in train_data:
            question_ls = format_article(i)
            writer.writerows(question_ls)

    with open(validation_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question'])

        for i in validation_data:
            question_ls = format_article(i)
            writer.writerows(question_ls)