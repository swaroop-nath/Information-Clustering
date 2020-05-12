import os
for i in range(1000):
    path = 'topic_' + str(i)
    for j in range(15):
        filename = 'doc_' + str(j)
        with open(os.path.join(path, filename), 'w+') as file:
            pass
    filename = 'wiki_article'
    with open(os.path.join(path, filename), 'w+') as file:
        pass
    filename = 'wiki_sum'
    with open(os.path.join(path, filename), 'w+') as file:
        pass