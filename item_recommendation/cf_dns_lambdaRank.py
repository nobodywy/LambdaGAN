import tensorflow as tf
from dis_model_dns_lambdaRank import DIS
import pickle
import numpy as np
import multiprocessing
import os
import math
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

cores = multiprocessing.cpu_count() - 1

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 16
USER_NUM = 943
ITEM_NUM = 1683
DNS_K = 5
BATCH_SIZE = 64
all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'
DIS_MODEL_FILE = workdir + "model_dns.pkl"
#########################################################################################
# Load data
#########################################################################################
user_pos_train = {}
with open(workdir + 'movielens-100k-train.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 0.1: #0 or 3.99
            if uid in user_pos_train:
                user_pos_train[uid].append(iid)
            else:
                user_pos_train[uid] = [iid]

user_pos_test = {}
with open(workdir + 'movielens-100k-test.txt')as fin:
    for line in fin:
        line = line.split()
        uid = int(line[0])
        iid = int(line[1])
        r = float(line[2])
        if r > 3.99:
            if uid in user_pos_test:
                user_pos_test[uid].append(iid)
            else:
                user_pos_test[uid] = [iid]

#all_users = user_pos_train.keys()
#all_users.sort()
all_users = sorted(user_pos_train)

def generate_dns(sess, model, filename): #判别器动态生成负样本 id
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(all_items - set(pos))

        for _ in range(len(pos)):
            choice = np.random.choice(candidates, DNS_K) #随机选取5个负样本item，对每个正样本item都选取一个分值最高的负样本
            choice_score = all_rating[choice]
            neg.append(choice[np.argmax(choice_score)])

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))  # uid pos_item_id neg_item_id

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def simple_test_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items - set(user_pos_train[u]))
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])

    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def simple_train_one_user(x):
    rating = x[0]
    u = x[1]

    test_items = list(all_items)
    item_score = []
    for i in test_items:
        item_score.append((i, rating[i]))

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_train[u]:
            r.append(1)
        else:
            r.append(0)

    p_3 = np.mean(r[:3])
    p_5 = np.mean(r[:5])
    p_10 = np.mean(r[:10])
    ndcg_3 = ndcg_at_k(r, 3)
    ndcg_5 = ndcg_at_k(r, 5)
    ndcg_10 = ndcg_at_k(r, 10)

    return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])

def simple_test(sess, model):
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    #test_users = user_pos_test.keys()
    test_users = list(user_pos_test.keys())  #edited
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret

def simple_train(sess, model):  # metric for train set
    result = np.array([0.] * 6)
    pool = multiprocessing.Pool(cores)
    batch_size = 128
    #test_users = user_pos_test.keys()
    test_users = list(user_pos_train.keys())  #edited
    test_user_num = len(test_users)
    index = 0
    while True:
        if index >= test_user_num:
            break
        user_batch = test_users[index:index + batch_size]
        index += batch_size

        user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
        user_batch_rating_uid = zip(user_batch_rating, user_batch)
        batch_result = pool.map(simple_train_one_user, user_batch_rating_uid)
        for re in batch_result:
            result += re

    pool.close()
    ret = result / test_user_num
    ret = list(ret)
    return ret


def generate_uniform(filename):
    data = []
    print ('uniform negative sampling...')
    for u in user_pos_train:
        pos = user_pos_train[u]
        candidates = list(all_items - set(pos))
        neg = np.random.choice(candidates, len(pos))
        pos = np.array(pos)

        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))

def main():  #首先初始化dis_dns判别器，使用判别器生成负样本作为判别器训练数据，以此更新dis_dns判别器;目的是预训练得到三个参数，bias，u v矩阵。
    np.random.seed(70)
    param = None
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.1)


    ### compute IDCG,DCG
    DCG = [] #save dcg for each rank
    #r = 0
    IDCG = [] #save idcg for each user
    idcg = 0
    for i in range(ITEM_NUM):
        DCG.append(1 / math.log(i + 2, 2))

    for i in range(USER_NUM):
        pos_len = len(user_pos_train[i])
        for j in range(pos_len):
            idcg += 1 / math.log(j + 2, 2)
        IDCG.append(idcg)
    ####

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    dis_log = open(workdir + 'dis_log_dns_lambdaRank.txt', 'w')
    print ("dis ", simple_test(sess, discriminator))
    best_p5 = 0.

    # generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples

    for epoch in range(150):
        generate_dns(sess, discriminator, DIS_TRAIN_FILE)  # dynamic negative sample  生成判别器的训练样本
        with open(DIS_TRAIN_FILE)as fin:
            for line in fin:
                line = line.split()
                u = int(line[0])
                i = int(line[1])
                j = int(line[2])
                rating = sess.run(discriminator.all_logits, {discriminator.u: [u]})
                o = pd.Series(rating)
                o.rank(ascending=False)
                rank_pos = int(o[i])
                rank_neg = int(o[j])
                delta_ndcg = abs(IDCG[rank_pos-1] - IDCG[rank_neg-1]) / IDCG[u]

                _ = sess.run(discriminator.d_updates,
                             feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                        discriminator.neg: [j], discriminator.delta_ndcg: [delta_ndcg]})

        result = simple_test(sess, discriminator)
        result_train = simple_train(sess, discriminator)
        print("epoch for test: " + str(epoch), "dis: ", result)
        print("epoch for train: " + str(epoch), "dis: ", result_train)
        if result[1] > best_p5:
            best_p5 = result[1]
            discriminator.save_model(sess, DIS_MODEL_FILE)
            print("best P@5 for test: ", best_p5)
            dis_log.write("best P@5 for test: " + '\t' + str(best_p5) + '\t')

        buf = '\t'.join([str(x) for x in result])
        buf_train = '\t'.join([str(x) for x in result_train])
        dis_log.write('test: ' + str(epoch) + '\t' + buf + '\n')
        dis_log.write('train: ' + str(epoch) + '\t' + buf_train + '\n')
        dis_log.flush()

    dis_log.close()


if __name__ == '__main__':
    main()
