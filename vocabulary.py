f = open("../data/sentences.train", 'r')
vocabulary = {}

line = f.readline()
while line:

	words = line.strip().split(' ')
	if len(words) <= 28:
		for word in words:
			if word not in vocabulary:
				vocabulary[word] = 0;
			vocabulary[word] += 1;

	line = f.readline()

f.close()

vocabulary["<bos>"] = 9999999
vocabulary["<eos>"] = 9999998
vocabulary["<pad>"] = 9999997
vocabulary["<unk>"] = 9999996

vocabulary = sorted(vocabulary.items(), key = lambda d: d[1], reverse = True)

out = open("vocabulary.txt", 'w')
for i in range(1000):
	out.write(vocabulary[i][0] + "\n")
out.close()
