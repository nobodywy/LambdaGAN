import pandas as pd
rat = [23,1,23,121,500]
o = pd.Series(rat)
c = o.rank(ascending=False)
print(c[1])
print()