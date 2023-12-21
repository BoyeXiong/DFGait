import os.path as osp
import time
import os
import threading
import itertools
import numpy as np
import pickle
import argparse


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--input_path', default='/8T/xiongjianbo/dataset/2D_Silhouettes', type=str,
                    help='Root path of raw dataset.')
parser.add_argument('--output_path', default='/data/xjb/database/gait3D/Gait3D-pkls/Gait3D_ske_pkl', type=str,
                    help='Root path for output.')
opt = parser.parse_args()


def get_pickle(thread_id, id_list, save_dir):
    for id in sorted(id_list):
        #print(f"Process threadID-PID: {thread_id}-{id}")
        cam_list = os.listdir(osp.join(data_dir, id))
        cam_list.sort()
        seqNum = 0
        for cam in cam_list:
            seq_list = os.listdir(osp.join(data_dir, id, cam))
            seqNum += len(seq_list)
        if seqNum > 40:
            print(f"id-seqNum: {id}-{seqNum}")
    



if __name__ == '__main__':

    data_dir = opt.input_path

    save_dir = opt.output_path

    start_time = time.time()
    maxnum_thread = 1

    all_ids = sorted(os.listdir(data_dir))
    num_ids = len(all_ids)

    proces = []
    for thread_id in range(maxnum_thread):
        indices = itertools.islice(range(num_ids), thread_id, num_ids, maxnum_thread)
        id_list = [all_ids[i] for i in indices]   # 多线程id列表
        thread_func = threading.Thread(target=get_pickle, args=(thread_id, id_list, save_dir))

        thread_func.start()
        proces.append(thread_func)

    for proc in proces:
        proc.join()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600,
        (time_elapsed - (time_elapsed // 3600) * 3600) // 60,
        time_elapsed % 60))