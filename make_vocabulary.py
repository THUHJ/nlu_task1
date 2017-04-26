# nlu_task1
f = open("../data/sentences.train",'r')
line = f.readline()
other = [',','.','!','?','`','\'\'']
vocabulary = {}
while line:
	
	line = line.strip()

	for i in other:
		line = line.replace(i,'')
	line = line.replace(' \'','\'')
	line = line.replace(' n\'','n\'')
	line = line.replace(' \'',' ')
	tmp = line.split(' ')
	for i in tmp:
		if len(i)==0:
			continue
		if i not in vocabulary:
			vocabulary[i]=0;
		vocabulary[i]+=1;

	line = f.readline()
vocabulary= sorted(vocabulary.items(), key=lambda d:d[1], reverse = True)
out = open("vocabulary.txt",'w')
for i in range(20000):
	
	out.write(vocabulary[i][0]+"\n")
	