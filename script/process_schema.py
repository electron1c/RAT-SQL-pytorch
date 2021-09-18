import copy
import json


def select_tables_and_process_schema(dataset):
    entities_map = {}
    with open(f'./data/{dataset}/db_schema.json', 'r') as f:
        db = json.load(f)
        for i, (t, c) in enumerate(db[0]['column_names']):
            db[0]['column_names'][i][1] = c.replace('(', '（').replace(')', '）').replace(' ', '_')

        for t, t1 in zip(db[0]['table_names'], db[0]['table_names_original']):
            entities_map[t1] = t
        for (t_id, c), (_, c1) in zip(db[0]['column_names'], db[0]['column_names_original']):
            if t_id == -1:
                entities_map[c1] = c
            else:
                t_name = db[0]['table_names'][t_id]
                t_name1 = db[0]['table_names_original'][t_id]
                entities_map[t_name1 + '@' + c1] = t_name + '@' + c
        db[0]['table_names_original'] = copy.deepcopy(db[0]['table_names'])
        db[0]['column_names_original'] = copy.deepcopy(db[0]['column_names'])

        with open(f'./data/{dataset}/db_schema_new.json', 'w') as ff:
            json.dump(db, ff, ensure_ascii=False, indent=2)

    with open(f'./data/{dataset}/db_content.json', 'r') as f:
        db = json.load(f)
        tables = {}
        for k, v in db[0]['tables'].items():
            for i, c in enumerate(v['header']):
                v['header'][i] = entities_map[k + '@' + c].split('@')[-1]
            tables[entities_map[k]] = v
        db[0]['tables'] = tables
        with open(f'./data/{dataset}/db_content_new.json', 'w') as ff:
            json.dump(db, ff, ensure_ascii=False, indent=2)

    new_map = {}
    for k, v in entities_map.items():
        new_map[k.lower()] = v.lower()

    with open(f'./data/{dataset}/entities_map.json', 'w') as f:
        json.dump(new_map, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    select_tables_and_process_schema('CSgSQL')
