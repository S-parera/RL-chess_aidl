class Prova:
    def __init__(self):
        self.reward = []

    def reset_ep(self):
        self.reward = []

class History:
    def __init__(self):
        self.episodes = []


episode = Prova()

history = History()

print(episode.reward)

episode.reward = [1,2,3,4]


history.episodes.append(episode)



episode = Prova()

episode.reward = [5,6,7,8]

history.episodes.append(episode)


for i in range(len(history.episodes)):

    print(history.episodes[i].reward)


l = [[1,2],[3,4]]

a =[]

for e in l:
    a+=e[:]

print(a)