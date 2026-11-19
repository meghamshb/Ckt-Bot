"""Existing assignment primitive only; no OCR/VLM or fallback implementation."""
def _minimum_assignment(costs):
    """Rectangular Hungarian minimum-cost matching; one distinct column per row."""
    if not costs: return [],0.0
    n=len(costs); m=len(costs[0])
    if n>m: raise ValueError('Assignment requires at least as many columns as rows')
    u=[0.]*(n+1);v=[0.]*(m+1);p=[0]*(m+1);way=[0]*(m+1)
    for i in range(1,n+1):
        p[0]=i; j0=0; minimum=[float('inf')]*(m+1); used=[False]*(m+1)
        while True:
            used[j0]=True; i0=p[j0];delta=float('inf');j1=0
            for j in range(1,m+1):
                if not used[j]:
                    cur=costs[i0-1][j-1]-u[i0]-v[j]
                    if cur<minimum[j]:minimum[j]=cur;way[j]=j0
                    if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    assigned=[None]*n
    for j in range(1,m+1):
        if p[j]:assigned[p[j]-1]=j-1
    return assigned,sum(costs[i][j] for i,j in enumerate(assigned))
