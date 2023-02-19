# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
def bisearch(target, start, end, arr): #재귀식 스탠다드

    if start > end :
        return None
    
    mid = (start+end) // 2
    
    if arr[mid] == target :
        return mid
    elif arr[mid] > target :
        end = mid -1
    else:
        start = mid +1
    
    return bisearch(target, start, end, arr)


nums = [i for i in range(0,10)]

# for i in nums :
#     bisearch(i,0,9,nums)
for i in nums :
    bisearch(i,0,9,nums)


# -

def bi(target, start, end, arr): # 반복식 스탠다드
    while start <= end :
        mid = (start+end) // 2
        
        if arr[mid] == target :
            return mid
        elif arr[mid] > target :
            end = mid - 1
        else :
            start = mid + 1
    return None


# +
def possible(length):
    
    cnt = 0
    
    for ch in churros:
        cnt += ch // length   
    
    if cnt >= K:              # mid로 나눠서 몫이 k 이상인애들 몫이 많다는건 mid가 타겟보다 작다는것 = 스타트점을 높여야함
        return True
    else:
        return False
        
    
    #length의 길이로, 모든 츄러스를 짤라봤을때
    # K개 이상이 되면 True, 안돼면 False


def search(start,end):
    answer = 0
    
    while start <= end:
        mid = (start + end) // 2
        
        if possible(mid):
            answer = mid
            start = mid + 1
        else :
            end = mid - 1
    return answer

N, K = map(int,input().split())

churros = []

for i in range(N):
    churros.append(int(input()))
    
answer = search(1, max(churros))
print(answer)
# -

lst=[1,2,3,4]
max(lst)


def test(n) :
    return n+1
res = test(3)
print(res)


# +
def func(start,end):
    #3번 반드시 주의!!
    #끝날 조건
    if start == end :
        print(num_list[start], end = " ")
        return
    
    # 들어가기전에 할 조건
    print(num_list[start], end = " ")
    
    #2번 -> 재귀 구성
    func(start + 1,end)
    
    #1번 -> 함수가 해야할일
    print(num_list[start], end = " ")



num_list = [3,5,4,6,2,9]
start, end = map(int,input().split())

func(start,end)

# -

6
3 5 4 6 2 9
0 5


# +
def func(start,end):
    
    print(start, end="")
    
    if start == end:
        return
    
    func(start + 1, N)
    func(start + 1, N)

N = int(input())

func(0, N)


# +
lst = [3,7,4,1,9,4,6,2]
def bumerang(start,end):
    if start == end:
        print(lst[start],end=' ')
        return
    
    print(lst[start],end=' ')
    bumerang(start-1,end)
    print(lst[start],end=' ')
    


n = 3
bumerang(n,0)
# -

1 4 7 3 7 4 1


def func(lvl):
#   print(lvl,end='')
  if lvl == level:
#     print(lvl,end='')    
    return
#   print(lvl,end='') # 리턴시엔 이건 안찍히고 다음재귀함수로 넘어감
  for _ in range(2):
    func(lvl+1)
#   print(lvl,end='') #리턴시에 마지막함수에 남은것들 찍히는거같츤데ㅐ
level = int(input())
func(0)

# +
branch, level = map(int,input().split())


cnt = 0

def func(lvl, branch):
    
    global cnt
    cnt += 1
    
    if lvl == level:
        return
    

    for _ in range(branch):
        func(lvl+1, branch)
        
func(0, branch)
print(cnt)

# +
cnt = 0

def func(lvl):
  global cnt
  cnt += 1
  if lvl == level :
    return
  for i in range(brunch):
    func(lvl+1)

brunch, level = map(int, input().split())
func(0)
print(cnt)

# +
from collections import deque

queue = deque()

queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)

print(queue)

queue.popleft()

print(queue)

queue.append(1)
queue.append(4)

queue.popleft()

print(queue)

# -

그래프는 G = (V,E) 볼텍스(정점)과 엣지(간선)의 집합 (컴퓨터공학에서의 그래프의 정의)
방향그래프 (화살표)
무방향그래프(간선의 방향이없ㅎ음)
차수(간선의 총개수) n의 차수(n한테 붙어있는 간선의 개수)
경로의 길이 (어떤 지점에 도달하기까지 간선의 개수)
사이클(어떤지점에 갔다가 다시 자신에게로 돌아오는경로)


# +
n,t = map(int, input().split())
dat = [[] for i in range(n+1)]
for i in range(t) :
  a,b = map(int, input().split())
  dat[a].append(b)

for i in range(n+1):
  if len(dat[i]) == 0 :
    continue
  print(f"{i} :", end=' ')
  for node in dat[i] :
    print(f"{node}",end=' ')
  print()

# +
n, t = map(int, input().split())

lst = [[],[],[],[],[],[],[]]
for i in range(t):
  a,b = map(int, input().split())
  lst[a].append(b)
  
for i in range(1,len(lst)):
  if len(lst[i]) == 0 :
    continue
  print(f"{i} :",end=' ')
  for j in lst[i] :
    print(j,end=' ')
  print()

# +
n = int(input())
adj = [ list(map(int,input().split())) for _ in range(n) ]
name = list(range(n))

answer = []

def dfs(now):
    global answer 
    
    answer.append(name[now])
    
    for i in range(n):
        if adj[now][i] == 1:
            dfs(i)


dfs(0)
print(answer)

# for i in range(len(answer)):
#     print(answer[i], end= " ")
# -

7
0 0 1 1 1 0 0 
0 0 0 0 0 0 0
0 0 0 0 0 1 1
0 1 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0

# +
lst = [[0,1,0,0,0],
       [0,0,1,1,0],
       [0,0,0,0,0],
       [0,0,0,0,0],
       [1,0,0,0,0]]

name = ['Amy','Bob','Chloe','Diane','Edger']
maxcnt = 0
for i in range(5):
  if lst[i].count(1) > maxcnt :
    maxcnt = lst[i].count(1)
    idx = i
print(name[idx])

# +
dat = [[0 for i in range(5)] for i in range(5)]

name = ['Amy', 'Bob', 'Chole', 'Diane', 'Edger']
dat[0][1] = 1
dat[1][2] = 1
dat[1][3] = 1
dat[4][0] = 1
maxcnt = 0
for i in range(len(dat)):
  if dat[i].count(1) > maxcnt :
    maxcnt = dat[i].count(1)
    idx = i
      
print(name[idx])

# +
lst=[[0,1,1,0,0,0,0,1],
     [0,0,0,0,0,0,0,0],
     [0,0,0,1,1,0,1,0],
     [0,0,0,0,0,1,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0]]
str = input()
name = ['A','B','C','D','E','F','G','H']
me = name.index(str)
flag = 0
for i in range(len(lst)) :
  if lst[i][me] == 1 :
    for j in range(len(lst)):
      if lst[i][j] == 1 and j != me :
        flag = 1
        print(name[j],end=' ')
if flag == 0 :
  print('없음')

      
