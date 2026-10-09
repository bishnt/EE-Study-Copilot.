import json

data = json.load(open('./data/syllabus.json'))
for s in data:
    marks = sum(c['marks'] for c in s['chapters'])
    print(f"{s['code']:9s} {s['subject']:4s} chapters={len(s['chapters']):2d} marks={marks}")
    assert 56 <= marks <= 64, 'marks should be close to 60'
print('subjects:', len(data))