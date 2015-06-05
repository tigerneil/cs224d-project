import sys
import csv
import StringIO
import json

# Usage: python process_dev.py <input_file> <output_file>

input_filename = sys.argv[1]
output_filename = sys.argv[2]

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



# Input: rel, subj_start, subj_end, subj_entity, obj_start, obj_end, obj_entity, words, lemmas, dependencies, sentence
# Output: JSON object
def process_line(line):
    strio = StringIO.StringIO(line)
    reader = csv.reader(strio, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        relation = str(mapping[row[0].replace("SLASH", "/")])
        subject_begin = row[1]
        subject_end = row[2]
        subject_entity = row[3]
        object_begin = row[4]
        object_end = row[5]
        object_entity = row[6]
        words = row[7]
        lemmas = row[8]
        dependencies_conll = row[9]
        gloss = row[10]
        output = [gloss, subject_entity, object_entity, subject_begin, subject_end, object_begin, object_end, relation]
        return json.dumps(output)


f_in = open(input_filename, 'r')
f_out = open(output_filename, 'w')

for line in f_in:
    if line and line != '':
        json_string = process_line(line)
        f_out.write(json_string)
        f_out.write('\n')

f_in.close()
f_out.close()