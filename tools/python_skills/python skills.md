# python skills

### Itertool functions

- combinations

```python
'''
itertools.combinations(iterable, r)
Return r length subsequences of elements from the input iterable
'''
from itertools import combinations
for i in combinations('ABCD',2):
    print(i)
```

results:

```python
('A', 'B')
('A', 'C')
('A', 'D')
('B', 'C')
('B', 'D')
('C', 'D')
```

- chain

```python
'''
chain(p, q,...),  return p0, p1, … plast, q0, q1, …
chain('ABC', 'DEF') --> A B C D E F
'''
from itertools import combinations,chain
for i in chain(['A',"B",1],[3,"H",2]):
    print(i)
```

results:

```
A B 1 3 H 2
```

- get all subsets

```python
from itertools import chain, combinations

def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i in range(len(arr))])

for i in subsets('ABC'):
    print(i)
```

return

```python
('A',)
('B',)
('C',)
('A', 'B')
('A', 'C')
('B', 'C')
('A', 'B', 'C')
```



### Numpy one-hot embedding

```python
def one_hot(t, class_num):
    I = np.eye(class_num).astype(np.long)
    res = I[t]   
    return res
```



























