
voca = {}
f=open("vocabulary.txt",'r')
line = f.readline()
t=0;
while line:
	voca[line.strip()] = t;
	t=t+1
	
	line  = f.readline()
f.close()
voca['<bos>']=t;
t=t+1;
voca['<eos>']=t;
t=t+1;
voca['<unk>']=t;
t=t+1;
voca['<pad>']=t;
t=t+1;


print ("begin to load")

f = open("../data/sentences.train",'r')
line = f.readline()
other = [',','.','!','?','`','\'\'']
vocabulary = {}
tt=0
while line:
	
	line = line.strip()

	for i in other:
		line = line.replace(i,'')
	line = line.replace(' \'','\'')
	line = line.replace(' n\'','n\'')
	line = line.replace(' \'',' ')
	tmp = line.split(' ')
	res=[]
	res.append(voca["<bos>"]);
	for i in tmp:
		if len(i)==0:
			continue;
		if i in voca:
			res.append(voca[i]);
		else:
			res.append(voca["<unk>"]);
	if (len(res)>=29):
		line = f.readline()
		continue
	while (len(res)<29):
		res.append(voca["<pad>"])
	res.append(voca["<eos>"])
	#print (res)
	tt+=1
	if tt%10000==0:
		print (tt)



	line = f.readline()
print (tt)
f.close()
	