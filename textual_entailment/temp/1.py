import collections

def zero(a=2):
    return a
bag = ['apple', 'orange', 'cherry', 'apple','apple', 'cherry', 'blueberry']
count = collections.defaultdict(zero)
for fruit in bag:
    count[fruit]
print(count)
# 输出：
# defaultdict(<class 'int'>, {'apple': 3, 'orange': 1, 'cherry': 2, 'blueberry': 1})


dic = collections.defaultdict((lambda :1))
dic['bbb']
print(dic)
# 输出：
# defaultdict(<function zero at 0x000001754EB4B488>, {'bbb': 0})