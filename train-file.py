vocabulary = {}
f = open("vocabulary.txt", 'r')
line = f.readline()
idx = 0
while line:
	vocabulary[line.strip()] = idx;
	idx += 1
	line  = f.readline()
f.close()

print("Start loading sentences ...")

f = open("../data/sentences.train", 'r')
out = open("sentences.code", 'w')
line = f.readline()
n = 0
while line:
	words = line.strip().split(' ')
	if (len(words) <= 28):
		code = [vocabulary["<bos>"]]
		for word in words:
			if word in vocabulary:
				code.append(vocabulary[word])
			else:
				code.append(vocabulary["<unk>"])
		while (len(code) < 29):
			code.append(vocabulary["<pad>"])
		code.append(vocabulary["<eos>"])
	assert len(code) == 30
	n += 1
	if (n % 10000 == 0):
		print(n)
	for c in code:
		out.write(str(code) + " ")
	out.write('\n')
	line = f.readline()
f.close()
