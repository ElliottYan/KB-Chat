import json
from nltk import wordpunct_tokenize as tokenizer
import argparse
import pickle
import os
import pdb

from utils.structure import Node

def entity_replace(temp, bot, user, names={}):
    # change poi, location, event first
    global_rp = {
        "pf changs": "p_.f_._changs",
        "p f changs": "p_.f_._changs",
        "'": "",
        "restaurants": "restaurant",
        "activities": "activity",
        "appointments": "appointment",
        "doctors": "doctor",
        "doctor s": "doctor",
        "optometrist s": "optometrist",
        "conferences": "conference",
        "meetings": "meeting",
        "labs": "lab",
        "stores": "store",
        "stops": "stop",
        "centers": "center",
        "garages": "garage",
        "stations": "station",
        "hospitals": "hospital"
    }

    for grp in global_rp.keys():
        bot = bot.replace(grp, global_rp[grp])
        user = user.replace(grp, global_rp[grp])

    # will it be not exactly match ?
    for name in names.keys():
        if name in bot:
            bot = bot.replace(name, names[name])
        if name in user:
            user = user.replace(name, names[name])

    for e in temp:
        for wo in e.keys():
            inde = bot.find(wo)
            if (inde != -1):
                bot = bot.replace(wo, e[wo])
            inde = user.find(wo)
            if (inde != -1):
                user = user.replace(wo, e[wo])
    return bot, user


def cleaner(token_array):
    new_token_array = []
    for idx, token in enumerate(token_array):
        temp = token
        if token == ".." or token == "." or token == "...": continue
        # remove exact time information ? only with am and pm ??
        if (token == "am" or token == "pm") and token_array[idx - 1].isdigit():
            new_token_array.pop()
            new_token_array.append(token_array[idx - 1] + token)
            continue
        if token == ":),": continue
        if token == "avenue": temp = "ave"
        if token == "heavey" and "traffic" in token_array[idx + 1]: temp = "heavy"
        if token == "heave" and "traffic" in token_array[idx + 1]: temp = "heavy"
        if token == "'": continue
        # ??
        if token == "-" and "0" in token_array[idx - 1]:
            new_token_array.pop()
            new_token_array.append(token_array[idx - 1] + "f")
            if "f" not in token_array[idx + 1]:
                token_array[idx + 1] = token_array[idx + 1] + "f"
        new_token_array.append(temp)
    return new_token_array


def main(root_path):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--json', dest='json',
                        default='kvret_train_public.json',
                        help='process json file')
    args = parser.parse_args()
    task = args.json.split('_')[1]

    with open(os.path.join(root_path, args.json)) as f:
        dialogues = json.load(f)

    with open(os.path.join(root_path, 'kvret_entities.json')) as f:
        entities_dict = json.load(f)

    # drop poi and poi_type here.
    global_kb_type = ['distance', 'traffic_info', 'location', 'weather_attribute', 'temperature', "weekly_time",
                      'event', 'time', 'date', 'party', 'room', 'agenda']
    global_temp = []
    di = {}
    # connect infos with '_' and map from original str to str with '_'
    for e in global_kb_type:
        for p in map(lambda x: str(x).lower(), entities_dict[e]):
            if "_" in p and p.replace("_", " ") != p:
                di[p.replace("_", " ")] = p
            else:
                if p != p.replace(" ", "_"):
                    di[p] = p.replace(" ", "_")
    global_temp.append(di)

    example_kbs = []

    for d in dialogues:
        roots = []

        if (d['scenario']['task']['intent'] == "navigate"):  # "schedule" "navigate"
            print("#navigate#")
            temp = []
            names = {}
            # iterate through all kb infos.
            for el in d['scenario']['kb']['items']:
                poi = " ".join(tokenizer(el['poi'].replace("'", " "))).replace(" ", "_").lower()
                slots = ['poi', 'distance', 'traffic_info', 'poi_type', 'address']
                # remvoe "'" and convert to lower
                for slot in slots:
                    el[slot] = " ".join(tokenizer(el[slot].replace("'", " "))).lower()
                names[el['poi']] = poi
                di = {
                    el['distance']: el['distance'].replace(" ", "_"),
                    el['traffic_info']: el['traffic_info'].replace(" ", "_"),
                    el['poi_type']: el['poi_type'].replace(" ", "_"),
                    el['address']: el['address'].replace(" ", "_"),
                }
                print(
                    "0 " + di[el['distance']] + " " + di[el['traffic_info']] + " " + di[el['poi_type']] + " poi " + poi)
                print("0 " + poi + " distance " + di[el['distance']])
                print("0 " + poi + " traffic_info " + di[el['traffic_info']])
                print("0 " + poi + " poi_type " + di[el['poi_type']])
                print("0 " + poi + " address " + di[el['address']])
                temp.append(di)

                # construct tree root for each kb item
                root = Node(poi, 'poi', layer=0)
                # except poi again
                for slot in slots[1:]:
                    root.children.append(Node(di[el[slot]], slot, layer=1))
                roots.append(root)

            # use for latter entity matching ?
            temp += global_temp

            # drop last one.
            if (len(d['dialogue']) % 2 != 0):
                d['dialogue'].pop()

            j = 1
            for i in range(0, len(d['dialogue']), 2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i + 1]['data']['utterance']).lower())))
                # replace entity names with names joined by "_"
                bot, user = entity_replace(temp, bot, user, names)
                navigation = global_kb_type  # ['distance','traffic_info']
                nav_poi = ['address', 'poi', 'type']
                gold_entity = []
                for key in bot.split(' '):
                    for e in navigation:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):
                            if (key == p):
                                gold_entity.append(key)
                            elif (key == str(p).replace(" ", "_")):
                                gold_entity.append(key)

                    for e in entities_dict['poi']:
                        for p in nav_poi:
                            if (key == str(e[p]).lower()):
                                gold_entity.append(key)
                            elif (key == str(e[p]).lower().replace(" ", "_")):
                                gold_entity.append(key)
                # gold entity for each turn of dialogue.
                gold_entity = list(set(gold_entity))
                if bot != "" and user != "":
                    print(str(j) + " " + user + '\t' + bot + '\t' + str(gold_entity))
                    j += 1
            print("")

        elif (d['scenario']['task']['intent'] == "weather"):  # "weather"
            print("#weather#")
            temp = []
            j = 1
            print("0 today " + d['scenario']['kb']['items'][0]["today"])
            today = d['scenario']['kb']['items'][0]["today"]
            for el in d['scenario']['kb']['items']:

                for el_key in el.keys():
                    el[el_key] = " ".join(tokenizer(el[el_key])).lower()
                loc = el['location'].replace(" ", "_")
                di = {el['location']: loc}
                temp.append(di)
                days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
                for day in days:
                    print("0 " + loc + " " + day + " " + el[day].split(',')[0].rstrip().replace(" ", "_"))
                    print("0 " + loc + " " + day + " " + el[day].split(',')[1].split(" ")[1] + " " +
                          el[day].split(',')[1].split(" ")[3])
                    print("0 " + loc + " " + day + " " + el[day].split(',')[2].split(" ")[1] + " " +
                          el[day].split(',')[2].split(" ")[3])

                # construct tree root for each kb item
                # root = Node(loc, 'location', layer=0)
                slots = ['weather', 'high', 'low']
                for day in days:
                    root = Node(loc, 'location', layer=0)
                    '''
                    tmp = Node(el[day], day, layer=1)
                    val = el[day]
                    splits = [item.strip() for item in val.split(',')]
                    tmp.children.append(Node(splits[0], 'weather', layer=2))
                    tmp.children.append(Node(splits[1], splits[1].split()[0], layer=2))
                    tmp.children.append(Node(splits[2], splits[2].split()[0], layer=2))
                    root.children.append(tmp)
                    '''
                    # change weather to 1-layer tree.
                    val = el[day]
                    splits = [item.strip() for item in val.split(',')]
                    root.children.append(Node(day, 'date', layer=1))
                    # more delicate for vals
                    root.children.append(Node(splits[1], splits[1].split()[0], layer=1))
                    root.children.append(Node(splits[2], splits[2].split()[0], layer=1))
                    # miss this in original dataset...
                    if today == day:
                        root.children.append(Node('yes', 'today', layer=1))
                    else:
                        root.children.append(Node('no', 'today', layer=1))

                    roots.append(root)

            temp += global_temp

            if (len(d['dialogue']) % 2 != 0):
                d['dialogue'].pop()

            for i in range(0, len(d['dialogue']), 2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i + 1]['data']['utterance']).lower())))
                bot, user = entity_replace(temp, bot, user)
                weather = global_kb_type  # ['location', 'weather_attribute','temperature',"weekly_time"]
                gold_entity = []
                for key in bot.split(' '):
                    for e in weather:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):
                            if (key == p):
                                gold_entity.append(key)
                            elif (key == str(p).replace(" ", "_")):
                                gold_entity.append(key)
                gold_entity = list(set(gold_entity))
                if bot != "" and user != "":
                    print(str(j) + " " + user + '\t' + bot + '\t' + str(gold_entity))
                    j += 1

            print("")

        if (d['scenario']['task']['intent'] == "schedule"):  # "schedule"
            print("#schedule#")
            temp = []
            names = {}
            j = 1
            # for all kb triple
            if (d['scenario']['kb']['items'] != None):
                for el in d['scenario']['kb']['items']:
                    for el_key in el.keys():
                        el[el_key] = " ".join(tokenizer(el[el_key])).lower()
                    ev = el['event'].replace(" ", "_")
                    names[el['event']] = ev
                    slots = ['time', 'date', 'party', 'room', 'agenda']
                    di = {}
                    for slot in slots:
                        if el[slot] == "-":
                            continue
                        if slot == "time":
                            print("0 " + ev + " " + slot + " " + el[slot].replace(" ", ""))
                            di[el[slot]] = el[slot].replace(" ", "")
                        else:
                            print("0 " + ev + " " + slot + " " + el[slot].replace(" ", "_"))
                            di[el[slot]] = el[slot].replace(" ", "_")
                    temp.append(di)

                    root = Node(ev, 'event', layer=0)
                    for slot in slots:
                        tmp = Node(el[slot], slot, layer=1)
                        root.children.append(tmp)

                    roots.append(root)

            temp += global_temp

            if (len(d['dialogue']) % 2 != 0):
                d['dialogue'].pop()

            for i in range(0, len(d['dialogue']), 2):
                user = " ".join(cleaner(tokenizer(str(d['dialogue'][i]['data']['utterance']).lower())))
                bot = " ".join(cleaner(tokenizer(str(d['dialogue'][i + 1]['data']['utterance']).lower())))
                bot, user = entity_replace(temp, bot, user, names)
                calendar = global_kb_type  # ['event','time', 'date', 'party', 'room', 'agenda']
                gold_entity = []
                for key in bot.split(' '):
                    for e in calendar:
                        for p in map(lambda x: str(x).lower(), entities_dict[e]):
                            if (key == p):
                                gold_entity.append(key)
                            elif (key == str(p).replace(" ", "_")):
                                gold_entity.append(key)
                gold_entity = list(set(gold_entity))
                if bot != "" and user != "":
                    print(str(j) + " " + user + '\t' + bot + '\t' + str(gold_entity))
                    j += 1

            print("")
        # add to example kbs.
        example_kbs.append(roots)

    # next step : save to file.
    with open(os.path.join(root_path, '{}_example_kbs.dat'.format(task)), 'wb') as f:
        pickle.dump(example_kbs, f)
