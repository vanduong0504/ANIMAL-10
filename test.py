

best_acc = 0
for intent_ac in [2,1,3,1.5,10]:
    if intent_ac > best_acc:
        best_acc = intent_ac
        improved_acc = 'Current accuracy {}, {}'.format(intent_ac, best_acc)
        print(improved_acc)