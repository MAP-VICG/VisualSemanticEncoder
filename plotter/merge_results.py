import json

# result_files = ['classification_results_000.json', 'classification_results_010.json',
#                 'classification_results_020.json', 'classification_results_030.json',
#                 'classification_results_040.json', 'classification_results_050.json',
#                 'classification_results_060.json', 'classification_results_070.json',
#                 'classification_results_080.json', 'classification_results_090.json',
#                 'classification_results_100.json']
#
# for json_file in result_files:
#     with open('/Users/damaresresende/Desktop/Results/' + json_file) as f:
#         new_data = json.load(f)
#
#     with open('../results/' + json_file) as f:
#         old_data = json.load(f)
#
#     for key in new_data.keys():
#         for sub_key in new_data[key].keys():
#             old_data[key][sub_key] = new_data[key][sub_key]
#
#     with open('../results_new/' + json_file, 'w+') as f:
#         json.dump(old_data, f, indent=4, sort_keys=True)


result_files = ['classification_results_000.json', 'classification_results_010.json',
                'classification_results_020.json', 'classification_results_030.json',
                'classification_results_040.json', 'classification_results_050.json',
                'classification_results_060.json', 'classification_results_070.json',
                'classification_results_080.json', 'classification_results_090.json',
                'classification_results_100.json']

for json_file in result_files:
    with open('/Users/damaresresende/Desktop/Results_CAT/' + json_file) as f:
        new_data = json.load(f)

    with open('../results/' + json_file) as f:
        old_data = json.load(f)

    for key in new_data.keys():
        for sub_key in new_data[key].keys():
            old_data[key][sub_key + '_v2'] = new_data[key][sub_key]

    with open('../results/' + json_file, 'w+') as f:
        json.dump(old_data, f, indent=4, sort_keys=True)
