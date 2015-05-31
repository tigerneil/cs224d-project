import sys
import csv
import StringIO
import json

mapping = {
    "per:country_of_death" : 0,
    "per:schools_attended" : 1,
    "per:other_family" : 2,
    "per:city_of_birth" : 3,
    "org:top_members/employees" : 4,
    "org:founded_by" : 5,
    "per:stateorprovinces_of_residence" : 6,
    "per:parents" : 7,
    "per:stateorprovince_of_death" : 8,
    "org:website" : 9,
    "per:stateorprovince_of_birth" : 10,
    "org:political/religious_affiliation" : 11,
    "per:age" : 12,
    "per:date_of_birth" : 13,
    "per:title" : 14,
    "per:member_of" : 15,
    "org:members" : 16,
    "org:city_of_headquarters" : 17,
    "per:origin" : 18,
    "per:alternate_names" : 19,
    "per:date_of_death" : 20,
    "per:children" : 21,
    "org:stateorprovince_of_headquarters" : 22,
    "org:member_of" : 23,
    "org:subsidiaries" : 24,
    "org:alternate_names" : 25,
    "per:religion" : 26,
    "per:spouse" : 27,
    "per:siblings" : 28,
    "per:cities_of_residence" : 29,
    "per:countries_of_residence" : 30,
    "org:country_of_headquarters" : 31,
    "org:number_of_employees/members" : 32,
    "per:cause_of_death" : 33,
    "per:charges" : 34,
    "org:shareholders" : 35,
    "per:country_of_birth" : 36,
    "per:employee_of" : 37,
    "org:dissolved" : 38,
    "org:parents" : 39,
    "org:founded" : 40,
    "per:city_of_death" : 41
}


# Process a single line of TSV and print result to stdout
def process_line(line):
    strio = StringIO.StringIO(line)
    reader = csv.reader(strio, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        gloss = row[0]
        dependencies_conll = row[1]
        words = row[2]
        lemmas = row[3]
        pos_tags = row[4]
        ner_tags = row[5]
        subject_id = row[6]
        subject_entity = row[7]
        subject_link_score = row[8]
        subject_ner = row[9]
        object_id = row[10]
        object_entity = row[11]
        object_link_score = row[12]
        object_ner = row[13]
        subject_begin = int(row[14])
        subject_end = int(row[15])
        object_begin = int(row[16])
        object_end = int(row[17])

        known_relations = None
        incompatible_relations = None
        annotated_relation = None

        # training
        if len(row) > 18:
            known_relations = row[18]
            incompatible_relations = row[19]
            annotated_relation = row[20]

        
        relations = ''
        if len(row) > 18:
            known_relations = known_relations[1:-1]
            known_relations_list = known_relations.split(',')
            known_relations_list = [str(mapping[x]) for x in known_relations_list]
            relations = ','.join(known_relations_list) # 0,2,3 or something like that

        words = words[1:-1]
        words = words.replace('\",\"', '~^~COMMA~^~')
        words = words.split(",")
        
        m1_begin = 0
        m2_begin = 0
        temp = []

        # Do we want to replace linked mention words ("Barack Obama") with their entity name (e.g. BarackObama)?
        replace_mention_words_with_entity_str = False

        for i, word in enumerate(words):
            cur = word

            if replace_mention_words_with_entity_str:
                if i == subject_begin:
                    cur = subject_entity
                    m1_begin = len(temp)
                if i == object_begin:
                    cur = object_entity
                    m2_begin = len(temp)

                if i > subject_begin and i < subject_end:
                    continue
                if i > object_begin and i < object_end:
                    continue

            if cur == '~^~COMMA~^~':
                cur = ','

            temp.append(cur)
            
        new_gloss = ' '.join(temp)

        if replace_mention_words_with_entity_str:
            # indices into the new gloss
            ind1 = min(m1_begin, m2_begin)
            ind2 = max(m1_begin, m2_begin)

            output = [new_gloss, str(ind1), str(ind2), relations]
            #print json.dumps(output)
            sys.stdout.write(json.dumps(output)) # maybe this will fix broken pipe error
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            output = [new_gloss, subject_entity, object_entity, str(subject_begin), str(subject_end), str(object_begin), str(object_end), relations]
            sys.stdout.write(json.dumps(output)) # maybe this will fix broken pipe error
            sys.stdout.write("\n")
            sys.stdout.flush()

for line in sys.stdin:
    line = line.strip()
    if line and line != '':
        process_line(line)
