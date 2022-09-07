arr = []
k = 0
with open("metadata.csv", 'r', encoding="utf-8") as f:
    for line in f:
        if not line in arr:
            arr.append(line)

print(len(arr))
arr.sort()
with open("metadata2.csv", 'w', encoding="utf-8") as f:
    for line in arr:
        f.write(line)


from phonemizer import phonemize as pho 
valid_ch = set(' !(),-.[]؟ءآأؤئابتثجحخدذرزسشصضطظعغفقكلمنهويًپچژکگۀی‌‍‎ ')

def clean(line):
    line = line.replace('؟',' ? ')
    line = line.replace('.',' . ')
    line = line.replace(',',' , ')
    line = line.replace('،',' , ')
    line = line.replace('!',' ! ')
    line = line.replace('¬',"")
    return line
k = 0
with open("metadata2.csv", 'r', encoding="utf-8") as f:
    with open("metadata3.csv", 'w', encoding="utf-8") as s:
        for line in f:
            dont_write = False
            line_list=line.split('|')
            line_list[1] = clean(line_list[1])
            # for ch in line_list[1]:
            # 	if not set(line_list[1]) <= valid_ch:
            # 		dont_write = True
            if not dont_write: 
                s.write('{}|{}\n'.format(line_list[0],' '.join(pho(line_list[1],language='fa', backend='espeak', preserve_punctuation=True).split())))
            k+= 1 
            if k%100 == 0:
                print(k)

