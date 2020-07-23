import json

crv = json.load(open('../data/squad/coref_data/coref_resolved_validation.json', 'r'))

keys = list(crv.keys())

i = 753
data = crv[keys[i]]

print('Original:')
print(data['document'])
print('\n\n')
print('Resolved:')
print(data['resolved'])
print('\n\n')

for cluster in data['clusters']:
   cluster_strs = list(map(lambda x: ' '.join(data['document'][x[0]: x[1] + 1]), cluster))
   print(' | '.join(cluster_strs))
