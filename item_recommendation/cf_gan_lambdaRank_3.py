import tensorflow as tf
from dis_model_lambdaRank import DIS
from gen_model_pairwise import GEN
import pickle
import numpy as np
import utils as ut
import multiprocessing
import os
import copy
import pandas as pd
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


cores = multiprocessing.cpu_count() - 1

#########################################################################################
# Hyper-parameters
#########################################################################################
EMB_DIM = 16
USER_NUM = 943
ITEM_NUM = 1683
BATCH_SIZE = 32
INIT_DELTA = 0.05

all_items = set(range(ITEM_NUM))
workdir = 'ml-100k/'
DIS_TRAIN_FILE = workdir + 'dis-train.txt'

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
        if r > 3.99:
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

    item_score = sorted(item_score, key=lambda x: x[1])
    item_score.reverse()
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

def generate_for_d(sess, model, filename):  # pairwise对 
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        pos_set = set(pos)

        rating = sess.run(model.all_rating, {model.u: [u]})
        rating = np.array(rating[0]) / 0.2  # Temperature
        exp_rating = np.exp(rating)
        prob = exp_rating / np.sum(exp_rating)

        neg_list = []
        for i in range(len(pos)):
            while True:
                neg = np.random.choice(np.arange(ITEM_NUM), p=prob)
                if neg not in pos_set:
                    neg_list.append(neg)
                    break
        for i in range(len(pos)):
            data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg_list[i]))

    with open(filename, 'w')as fout:
        fout.write('\n'.join(data))


def main():
    print ("load model...")
    param = pickle.load(open(workdir + "model_dns_TI.pkl",'rb'),encoding='bytes')
    #pickle.loa
    generator = GEN(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.0 / BATCH_SIZE, param=param, initdelta=INIT_DELTA,
                    learning_rate=0.001)
    discriminator = DIS(ITEM_NUM, USER_NUM, EMB_DIM, lamda=0.1 / BATCH_SIZE, param=None, initdelta=INIT_DELTA,
                        learning_rate=0.001)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    print ("gen ", simple_test(sess, generator))
    print ("dis ", simple_test(sess, discriminator))

    dis_log = open(workdir + 'dis_lambdaRank_v3_log.txt', 'w')
    gen_log = open(workdir + 'gen_lambdaRank_v3_log.txt', 'w')

    TI = []
    ti = 0
    for i in range(ITEM_NUM):
        ti += 1 / (i + 1)
        TI.append(ti)

    # minimax training
    best = 0.
    for epoch in range(200):
        if epoch >= 0:
            for d_epoch in range(4): #100
                if d_epoch % 5 == 0:
                    generate_for_d(sess, generator, DIS_TRAIN_FILE)
                    train_size = ut.file_len(DIS_TRAIN_FILE)
                index = 1
                while True:
                    if index > train_size:
                        break
                    if index + BATCH_SIZE <= train_size + 1:
                        input_user, input_item_pos, input_item_neg = ut.get_batch_data_pairwise(DIS_TRAIN_FILE, index, BATCH_SIZE)
                    else:
                        input_user, input_item_pos, input_item_neg = ut.get_batch_data_pairwise(DIS_TRAIN_FILE, index,
                                                                                train_size - index + 1)
                    index += BATCH_SIZE

                    #delta NDCG
                    delta_ndcg_list = []
                    former_user_id = -1
                    former_user_rating = []
                    for i in range(len(input_user)):
                        if (input_user[i] != former_user_id):
                            rating = sess.run(discriminator.all_logits, {discriminator.u: input_user[i]})
                            former_user_id = input_user[i]
                            former_user_rating = rating
                        else:
                            rating = former_user_rating
                        rating = list(rating)
                        o = pd.Series(rating)
                        o = o.rank(ascending=False)
                        rank_pos = int(o[input_item_pos[i]])
                        delta_ndcg = TI[rank_pos - 1] / TI[ITEM_NUM - 1]
                        delta_ndcg = pow(1.5, delta_ndcg)
                        delta_ndcg_list.append(delta_ndcg)

                    _ = sess.run(discriminator.d_updates,
                                 feed_dict={discriminator.u: input_user, discriminator.i_pos: input_item_pos,
                                            discriminator.i_neg: input_item_neg, discriminator.delta_ndcg: delta_ndcg_list})
                print('d_epoch: ',d_epoch)
            # Train G
            for g_epoch in range(2):  # 50
                for u in user_pos_train:
                    sample_lambda = 0.2
                    pos = user_pos_train[u]
                    pos_set = set(pos)

                    rating = sess.run(generator.all_logits, {generator.u: u})
                    exp_rating = np.exp(rating)
                    prob = exp_rating / np.sum(exp_rating)  # prob is generator distribution p_\theta

                    pn = (1 - sample_lambda) * prob
                    pn[pos] += sample_lambda * 1.0 / len(pos)
                    # Now, pn is the Pn in importance sampling, prob is generator distribution p_\theta
                    # generate pos，neg pair
                    '''
                    sample_list = []
                    for i in range(len(pos)):
                        while True:
                            sample = np.random.choice(np.arange(ITEM_NUM), p=pn)
                            if sample not in pos_set:
                                sample_list.append(sample)
                                break
                    sample = sample_list
                    '''
                    sample = np.random.choice(np.arange(ITEM_NUM), len(pos), p=pn)

                    #compute delta_ndcg
                    # delta NDCG
                    delta_ndcg_list = []
                    rating = list(rating)
                    o = pd.Series(rating)
                    o = o.rank(ascending=False)
                    for i in range(len(pos)):
                        rank_pos = int(o[pos[i]])
                        delta_ndcg = TI[rank_pos - 1] / TI[ITEM_NUM - 1]
                        delta_ndcg = pow(1.5, delta_ndcg)
                        delta_ndcg_list.append(delta_ndcg)

                    ###########################################################################
                    # Get reward and adapt it with importance sampling
                    ###########################################################################
                    reward = sess.run(discriminator.reward, {discriminator.u: u, discriminator.i_pos: pos,
                                                             discriminator.i_neg: sample, discriminator.delta_ndcg: delta_ndcg_list})
                    reward = reward * prob[sample] / pn[sample]
                    ###########################################################################
                    # Update G
                    ###########################################################################
                    _ = sess.run(generator.gan_updates,
                                 {generator.u: u, generator.i: sample, generator.reward: reward})

                result = simple_test(sess, generator)
                result_train = simple_train(sess, generator)
                print("epoch ", epoch, "gen——test: ", result)
                print("epoch ", epoch, "gen——train: ", result_train)
                buf = '\t'.join([str(x) for x in result])
                buf_train = '\t'.join([str(x) for x in result_train])
                gen_log.write('test : ' + str(epoch) + '\t' + buf + '\n')
                gen_log.write('train : ' + str(epoch) + '\t' + buf_train + '\n')
                gen_log.flush()

                p_5 = result[1]
                if p_5 > best:
                    print('best: ', result)
                    gen_log.write('best: ' + str(epoch) + '\t' + buf + '\n')
                    gen_log.flush()
                    best = p_5
                    generator.save_model(sess, "ml-100k/gan_generator_lambdaRank.pkl")
    gen_log.close()
    dis_log.close()


if __name__ == '__main__':
    main()
