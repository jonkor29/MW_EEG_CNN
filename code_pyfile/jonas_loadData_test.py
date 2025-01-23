import loadData1 as ld1
import numpy as np

study = 1
input_type = 'raw' 
conds = ['ot', 'mw']
tasks = ['sart', 'vs']  # dual tasks for variability (default)
win_lens = 12 #[[12,12], [10,20]]  # load data from x seconds before each probe
subs = range(1,31)  # 1-30 for study 1, 301-330 for study 2
channels = range(32)
gpu2use = -1 #cpu
#tpx, tpy, tps = ld1.load_dataset_n([subs[0]], tasks[0], [input_type], conds, nBacks=3, norm=False, sidx='all', gpu2use=gpu2use)


#xs_train, xs_test, ys_train, ys_test = ld1.load_dataset_n_split(subs, tasks[1], [input_type], conds, nBacks=3, norm=False, testSize=0.2, randSeed=42, sidx='all', gpu2use=gpu2use)

if input('Do you want to create and save a new dataset? (y/n)') == 'y':
    xs, ys, ss = ld1.load_dataset_n_tasks(subs, tasks, [input_type], conds, winlens=10, norm=False, sidx='all', gpu2use=gpu2use)
    #xs, ys, ss = ld1.load_dataset_n_tasks([subs[0]], [tasks[0]], [input_type], conds, winlens=6, norm=False, sidx='all', gpu2use=gpu2use)
    data = {}
    data['X'] = xs
    data['y'] = ys
    data['s'] = ss
    np.savez('loaded_dataset.npz',**data)

if input('Do you want to create and save a new dataset subject and task-wise? (y/n)') == 'y':
    data = {}
    data['X'] = {}
    data['y'] = {}
    data['s'] = {}
    for sub in subs:
        data['X'][sub] = {}
        data['y'][sub] = {}
        data['s'][sub] = {}
        for task in tasks:
            xs, ys, ss = ld1.load_dataset_n_tasks([sub], [task], [input_type], conds, winlens=10, norm=False, sidx='all', gpu2use=gpu2use)
            data['X'][sub][task] = xs
            data['y'][sub][task] = ys
            data['s'][sub][task] = ss
    np.savez('loaded_dataset_subject_task_wise.npz',**data)

#x_all, y_all = ld1.load_dataset(1, 'vs', 'raw', ['ot', 'mw'], 3)

if input('Do you want to load "loaded_dataset" and display info about it? (y/n)') == 'y':
    data = np.load('loaded_dataset_subject_task_wise.npz', allow_pickle=True)
    X_dict = data['X'].item()
    y_dict = data['y'].item()
    s_dict = data['s'].item()

    for sub in subs:
        for task in tasks:
            print(f"data['X'][{sub}][{task}].shape:", X_dict[sub][task].shape)
            print(f"data['y'][{sub}][{task}].shape:", y_dict[sub][task].shape)
            print(f"data['s'][{sub}][{task}].shape:", len(s_dict[sub][task]))
            assert X_dict[sub][task].shape[0] == y_dict[sub][task].shape[0] == len(s_dict[sub][task])
            assert X_dict[sub][task].shape[1] == 32
            assert X_dict[sub][task].shape[2] == 180
            assert s_dict[sub][task][0] == sub




""" print("--------------------------------")
print("x_all.shape:", x_all.shape)
print("y_all.shape:", y_all.shape)
print("y_all[:10]:", y_all[:10])

print("--------------------------------")
print("xs_train.shape:", xs_train.shape)
print("xs_test.shape:", xs_test.shape)
print("ys_train.shape:", ys_train.shape)
print("ys_test.shape:", ys_test.shape)
print("ys_test[:100]:", ys_test[:100]) 

print("--------------------------------")
print("xs.shape:", xs.shape)
print("ys.shape:", ys.shape)
print("len(ss):", len(ss))
"""


