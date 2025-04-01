def lamda_decay(lamda, ep):
    l = lamda
    if ep >= 100:
        l = lamda / 2
    elif ep > 200:
        l = lamda / 5
    return l
