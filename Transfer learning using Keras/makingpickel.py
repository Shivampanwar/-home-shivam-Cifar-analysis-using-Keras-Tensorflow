import pickle

with open("train.p", "rb") as f:
    w = pickle.load(f)

pickle.dump(w, open("train.pkl","wb"), protocol=2)
with open("val.p", "rb") as f:
    v = pickle.load(f)
pickle.dump(v, open("val.pkl","wb"), protocol=2)


