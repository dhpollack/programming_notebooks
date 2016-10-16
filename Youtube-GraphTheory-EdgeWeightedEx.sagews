︠2e450aa8-978f-4dd7-bef5-643f502926e8s︠
import numpy as np
a = [0, 3, 1, 2, 0, 0]
b = [3, 0, 0, 0, 0, 0]
c = [1, 0, 0, 0, 4, 2]
d = [2, 0, 0, 0, 1, 0]
e = [0, 0, 4, 1, 0, 0]
f = [0, 0, 2, 0, 0, 0]
A = Matrix([a, b, c, d, e, f])
G = Graph(A, weighted=True)

def dijkstras_algo(G, u, S = [], t = []):
    byw = G.weighted()
    if not S:
        # initial empty list
        S.append(u)
        t = np.zeros(G.order())
        t_max = 1 + np.sum(G.weighted_adjacency_matrix()) / 2
        for i in range(len(t)):
            if i != u:
                t[i] = t_max
    neighborhood = list(set(G.neighbors(u)) - set(S))
    for v in neighborhood:
        if v not in S:
            t[v] = min(t[v], t[u] + G.distance(u,v,by_weight=byw))
    left = list(set(G.vertices()) - set(S))
    w = left[np.argmin(t[left])]
    S.append(w)
    if set(G.vertices()) != set(S):
        dijkstras_algo(G, w, S, t)
    return S, t

S, t = dijkstras_algo(G, 0)
print S, t

︡9ffef417-fe72-4731-adbf-f9f2fd917223︡{"stdout":"[0, 2, 3, 1, 4, 5] [ 0.  3.  1.  2.  3.  3.]\n"}︡{"done":true}︡
︠2a091a46-11a3-4c7c-ae57-e523ef4d7dd4s︠
a = [0,2,3,0,0,0]
b = [2,0,2,1,3,3]
c = [3,2,0,0,1,0]
d = [0,1,0,0,2,1]
e = [0,3,1,2,0,2]
f = [0,3,0,1,2,0]
A = Matrix([a,b,c,d,e,f])
G = Graph(A, weighted=True)
S, t = dijkstras_algo(G, 0, S = [], t = [])
print S, t

︡8eaafbcd-1aeb-45de-a972-fc33646cade1︡{"stdout":"[0, 1, 2, 3, 4, 5] [ 0.  2.  3.  3.  4.  4.]\n"}︡{"done":true}︡
︠1e4eee2a-d583-4206-a288-aa6d2f21c0d2s︠
x = 4

a = [0, 3, 1, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0]
b = [3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
c = [1, 0, 0, 4, 0, 0, 7, 0, 0, 0, 0, 0, 0]
d = [0, 1, 4, 0, 0, 2, 0, 6, 0, 0, 0, 0, 0]
e = [5, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]
f = [0, 0, 0, 2, 2, 0, 1, x, 0, 0, 0, 1, 0]
g = [0, 0, 7, 0, 0, 1, 0, 2, 0, 6, 0, 0, 0]
h = [0, 0, 0, 6, 0, x, 2, 0, 1, 0, 0, 0, 0]
i = [0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 7, 0, 1]
j = [0, 0, 0, 0, 0, 0, 6, 0, 2, 0, 4, 0, 0]
k = [0, 0, 0, 0, 0, 0, 0, 0, 7, 4, 0, 3, 0]
l = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3, 0, 4]
m = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 4, 0]
A=Matrix([a, b, c, d, e, f, g, h, i, j, k, l, m])
G = Graph(A, weighted=True)

print G.shortest_path(0,9, by_weight=True)
print G.shortest_path(0,9, by_weight=False)
print G.distance(0,9, by_weight=True)
print G.distance(0,9, by_weight=False)

S, t = dijkstras_algo(G, 0, S = [], t = [])
print S, t
︡8fd014ff-0088-45a8-8475-cc59e2d270b0︡{"stdout":"[0, 1, 3, 5, 6, 7, 8, 9]\n"}︡{"stdout":"[0, 2, 6, 9]\n"}︡{"stdout":"12\n"}︡{"stdout":"3\n"}︡{"stdout":"[0, 2, 1, 3, 4, 5, 6, 11, 7, 8, 10, 12, 9] [  0.   3.   1.   4.   5.   6.   7.   9.  10.  12.  10.   7.  11.]\n"}︡{"done":true}︡
︠9ba6667e-18b2-40ba-9dc5-5123e35dd7d4s︠
a = [0,7,4,5,0,0,0,0,0]
b = [7,0,2,0,25,0,0,0,0]
c = [4,2,0,0,0,0,0,9,0]
d = [5,0,0,0,0,9,0,0,0]
e = [0,25,0,0,0,0,10,0,0]
f = [0,0,0,9,0,0,0,20,0]
g = [0,0,0,0,10,0,0,0,2]
h = [0,0,9,0,0,20,0,0,3]
i = [0,0,0,0,0,0,2,3,0]
A = Matrix([a,b,c,d,e,f,g,h,i])
G = Graph(A, weighted=True)
G.plot()

S, t = dijkstras_algo(G, 0, S = [], t = [])
print S, t
︡89a52e87-0c4f-4ba7-8c4f-a943ed0d913a︡{"file":{"filename":"/projects/e4d6c2ac-d4ac-4792-9987-c8fef33777ee/.sage/temp/compute1-us/12429/tmp_7Y2ieD.svg","show":true,"text":null,"uuid":"c5d4a60b-592f-4e49-9116-e3eadb1a9287"},"once":false}︡{"html":"<div align='center'></div>"}︡{"stdout":"[0, 2, 3, 1, 7, 5, 8, 6, 4] [  0.   6.   4.   5.  28.  14.  18.  13.  16.]\n"}︡{"done":true}︡









