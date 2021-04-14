# -*- coding: UTF-8 -*- 
import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

def length_clip(content, query, option):
    content = content.replace("\n", "\\n")
    query = query.replace("\n", "\\n")
    option = option.replace("\n", "\\n")
    return content[:383], query[:50], option[:75]


def format_article(article_json):
    content = article_json['Content']
    question_ls = []
    for i in article_json['Questions']:
        query = i['Question']

        ans = None
        if 'Answer' in i:
            ans = ord(i['Answer']) - ord('A')
        quesiton_id =i['Q_id']

#         for j, option in enumerate(i['Choices']):
        for j in range(4):
            option = i['Choices'][j] if j < len(i['Choices']) else "[PAD]"
            score = 1 if ans == j else 0

            content, query, option = length_clip(content, query, option)
            format_str = f"[CLS]{content}[SEP]{query}[SEP]{option}[SEP]"

            if ans is not None:
                question_ls.append((quesiton_id, format_str, score))
            else:
                question_ls.append((quesiton_id, format_str))
                

    return question_ls

def cdf_plot(cdf_ls, title, path):
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(cdf_ls)
    ax.set_title(title)

    plt.savefig(path)


def cut_pdf(pdf_ls):
    last_index = np.max(np.nonzero(pdf_ls))
    return pdf_ls[:last_index+1]

def length_statistic(article_ls):
    content_length_max = 0

    for article in article_ls:
        content_len = len(article['Content'])
        if content_length_max < content_len:
            content_length_max = content_len

    content_length_max += 1

    content_pdf = [0] * content_length_max
    query_pdf = [0] * content_length_max
    option_pdf = [0] * content_length_max

    for article in article_ls:
        content_len = len(article['Content'])
        content_pdf[content_len] += 1

        for quesion in article['Questions']:
            query_len = len(quesion['Question'])
            query_pdf[query_len] += 1

            for option in quesion['Choices']:
                option_len = len(option)
                option_pdf[option_len] += 1

    content_pdf = cut_pdf(content_pdf)
    query_pdf = cut_pdf(query_pdf)
    option_pdf = cut_pdf(option_pdf)

    content_cdf = np.cumsum(content_pdf)
    content_cdf = content_cdf / content_cdf[-1]

    query_cdf = np.cumsum(query_pdf)
    query_cdf = query_cdf / query_cdf[-1]

    option_cdf = np.cumsum(option_pdf)
    option_cdf = option_cdf / option_cdf[-1]

    cdf_plot(content_cdf, "Content", "plot/content_cdf.png")
    cdf_plot(query_cdf, "query", "plot/query_cdf.png")
    cdf_plot(option_cdf, "option", "plot/option_cdf.png")



def format_train():
    train_path = "data/train.json"
    train_csv_path = "data/train.csv"

    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question', 'score'])

        for i in train_data:
            question_ls = format_article(i)
            writer.writerows(question_ls)


def format_vali():
    validation_path = "data/validation.json"
    validation_csv_path = "data/validation.csv"

    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    with open(validation_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'question'])

        for i in validation_data:
            question_ls = format_article(i)
            writer.writerows(question_ls)


if __name__ == '__main__':
    format_train()
    format_vali()

    with open("data/train.json", 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    length_statistic(train_data)