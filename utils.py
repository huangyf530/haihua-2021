import json

train_keys = ["ID", "Q_id", "Content", "Question", "Choices", "Answer"]
test_keys = ["ID", "Q_id", "Content", "Question", "Choices"]


def convert_data_structure(data, ispredict=False):
    assert isinstance(data, list)
    new_data = {}
    if ispredict:
        keys = test_keys
    else:
        keys = train_keys
    for key in keys:
        new_data[key] = []
    for d in data:
        passage = d['Content']
        for q in d['Questions']:
            new_data["ID"].append(d['ID'])
            new_data["Content"].append(passage)
            new_data["Question"].append(q['Question'])
            new_data["Choices"].append(q['Choices'])
            new_data['Q_id'].append(q['Q_id'])
            if not ispredict:
                new_data["Answer"].append(q['Answer'])
    return new_data

