import random
random.seed(123)

dataset = [[1,'v'],[5,'b'],[7,'f'],[4,'h'],[0,'l']]
sample = random.sample(dataset, 3)

print(sample)
