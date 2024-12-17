import loadData1 as ld1


study = 1
input_type = 'raw' 
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 12 #[[12,12], [10,20]]  # load data from x seconds before each probe
subs = range(1,31)  # 1-30 for study 1, 301-330 for study 2
channels = range(32)
gpu2use = -1 #cpu
tpx, tpy, tps = ld1.load_dataset_n([subs[0]], tasks[0], [input_type], conds, nBacks=3, norm=False, sidx='all', gpu2use=gpu2use)


#xs_train, xs_test, ys_train, ys_test = ld1.load_dataset_n_split(subs, tasks[1], [input_type], conds, nBacks=3, norm=False, testSize=0.2, randSeed=42, sidx='all', gpu2use=gpu2use)

#xs, ys, ss = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlens=24, norm=False, sidx='all', gpu2use=gpu2use)
xs, ys, ss = ld1.load_dataset_n_tasks([subs[0]], [tasks[0]], [input_type], conds, winlens=6, norm=False, sidx='all', gpu2use=gpu2use)

#x_all, y_all = ld1.load_dataset(1, 'vs', 'raw', ['ot', 'mw'], 3)

""" print("--------------------------------")
print("x_all.shape:", x_all.shape)
print("y_all.shape:", y_all.shape)
print("y_all[:10]:", y_all[:10])

print("--------------------------------")
print("xs_train.shape:", xs_train.shape)
print("xs_test.shape:", xs_test.shape)
print("ys_train.shape:", ys_train.shape)
print("ys_test.shape:", ys_test.shape)
print("ys_test[:100]:", ys_test[:100]) """

print("--------------------------------")
print("xs.shape:", xs.shape)
print("ys.shape:", ys.shape)
print("len(ss):", len(ss))
print("ys:", ys)
print("ss:", ss)


