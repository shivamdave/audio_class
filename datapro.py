#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:10:20 2018

@author: shivam
"""
import multiprocessing as mp
import os
from os import path
from multiprocessing import Pool
from os import listdir, mkdir
from os.path import isdir, exists, join
from shutil import copy

import h5py
import librosa as l
import numpy as np
import pandas as pd

import sys
import pickle

import pdb
class DataProcess:

    def __init__(self):
        self.cpu = mp.cpu_count()

        self.a = 5

        self.sorted_audio_path = "/raid/dataset/sorted/"
        self.sorted_wavaudio_path = "/raid/dataset/wavsorted/"
        self.unsorted_audio_path = "/raid/dataset/fma_medium/"
        self.fma_metadata = "/raid/dataset/fma_metadata/"
        self.all_fma_medium = "/raid/dataset/all_fma_medium/"
        '''
        self.sorted_audio_path = "C:/Users/sdave/data/sorted/"
        self.sorted_wavaudio_path = "C:/Users/sdave/data/wavsorted/"
        self.unsorted_audio_path = "C:/Users/sdave/data/fma_medium/"
        self.fma_metadata = "C:/Users/sdave/data/fma_metadata/"
        self.all_fma_medium = "C:/Users/sdave/data/all_fma_medium/"
        '''
        self.f = os.path.join(self.sorted_audio_path, "testfile.hdf5")
        self.p = os.path.join(self.sorted_audio_path, "process.hdf5")
        self.g = os.path.join(self.sorted_audio_path, "genre_list.p")

        self.temprawfolder = np.array([])
        self.temploadfolder = np.array([])
        self.tempprofolder = np.array([])
        self.sec = 3
        self.df = pd.read_csv(filepath_or_buffer=self.fma_metadata + 'tracks.csv', header=None, usecols=[0, 40])

        self.dfmod = self.df.dropna(axis=0)
        self.dfmod = self.dfmod.reset_index(drop=True)
        self.dfmod[0] = self.dfmod[0].astype(int)
        self.dfmod = np.asarray(self.dfmod)
        self.Y_genre_strings = np.array([])
        self.data = np.array([])

    def merge_audio_files(self):
        if not exists(self.all_fma_medium):
            os.mkdir(self.all_fma_medium)
        else:
            pass

        #  below code copies the audio files to one directory i.e. self.all_fma_medium
        [(copy(self.unsorted_audio_path + i + "/" + j, self.all_fma_medium)) for i in listdir(self.unsorted_audio_path)
         if isdir(self.unsorted_audio_path + i + "/") for j in listdir(self.unsorted_audio_path + i)]

    def dataShape(self):
        with h5py.File(self.f, 'a') as dz:
            for i in list(dz.keys()):
                print(i)
                for j in dz[i]:
                    a = dz[i + '/' + j][:]
                    print(a.shape)

    def mpsorte(self, x):
        print('the value of ', x)
        afg = int(x[:-4])
        print(x, afg)

        match = np.where(self.dfmod == afg)[0]
        genre = self.dfmod[match]

        try:
            print('in try ', genre[0][1])
            musictype = genre[0][1]

        except IndexError:
            print('IndexError has occurred')

        print(x, musictype)
        if '/' in musictype:
            musictype = musictype.replace('/', '')

        if exists(self.sorted_audio_path + musictype + '/'):
            copy(self.all_fma_medium + x, self.sorted_audio_path + musictype + '/')

        else:
            mkdir(self.sorted_audio_path + musictype + '/')
            copy(self.all_fma_medium + x, self.sorted_audio_path + musictype + '/')

        '''
        we can try to use sync multiprocessing with below lines
        self.dfmod = np.delete(self.dfmod, genre[0][0], 0)
        self.dfmod = self.dfmod.reset_index(drop=True)
        '''
        return None

    def sorte(self):
        if not exists(self.sorted_audio_path):
            mkdir(self.sorted_audio_path)
            print('the folder doesnt exist')
            self.sorte()
        else:
            print('the folder does exist')
            # iter_audio_files =iter(listdir(self.all_fma_medium))
            iter_audio_files = listdir(self.all_fma_medium)
            pool = Pool(processes=self.cpu)
            pool.map_async(self.mpsorte, iter_audio_files)
            # results = [pool.apply_async(self.mpsorte, (listdir(self.all_fma_medium),))]
            # results = [pool.apply_async ( self.mpsorte, (cnt,ij,)) for cnt,ij in enumerate(listdir(self.all_fma_medium))]
            pool.close()
        # args=(listdir(self.all_fma_medium)
        return None

    def sort_count(self, num=0):
        [(name, len(os.listdir(self.sorted_audio_path + name))) for name in os.listdir(self.sorted_audio_path) if
         len(os.listdir(self.sorted_audio_path + name)) > num]  # for ij in self.sorted_audio_path for

    def loadfile(self, foldername):
        print('loading files path from folder {}'.format(foldername))

        self.temprawfolder = [join(self.sorted_audio_path + foldername, f) for f in
                              listdir(self.sorted_audio_path + foldername)]

        tempfol = {foldername: self.temprawfolder}
        return tempfol

    def stackker(self, musicfile):
        global testvar
        print((self.sorted_audio_path + self.musicpath + '/' + musicfile))
        try:
            y, sr = l.load(self.sorted_audio_path + self.musicpath + '/' + musicfile)
            se = self.sec
            y_len = len(y)
            tempstack = np.reshape(y[0:int(y_len/(sr * se)) * (sr * se):], (int(y_len / (sr * se)), (sr * se)))
            return tempstack
        except Exception as e:
            print('i am in EOFFILE in stackker return None')
            print('**', musicfile)
            return None

    def chopnload(self, tempfol):

        # tempvstack = np.array([])
        global folname, musicpath, tempstack
        folname = None
        self.musicpath = None
        tempstack = None

        for h, i in enumerate(tempfol.keys()):
            print(i)
            folname = i
            self.musicpath = i
            print(self.sorted_audio_path + self.musicpath)
        global jobs, dfg
        jobs = None
        dfg = []

        with Pool(processes = self.cpu) as p:
            # testvar = p.self.sorte,musicpath)
            jobs = (p.map(self.stackker, (listdir(self.sorted_audio_path + self.musicpath))))
            jobs = np.vstack(jobs[(isinstance(jobs,np.ndarray))])

            #job = [np.vstack(i) for i in jobs if isinstance(i,np.ndarray)]

            print('jobs:', jobs)

        # p = Process(target = sorte , musicfolder)
        #mode = 'r+' if  dt['/chopped/{}'.format(folname) in self.f else 'a'

        with h5py.File(self.f, 'a') as dt:
            print('in chopped chopping {}'.format(folname))
            try:
                dt.create_dataset('/chopped/{}'.format(folname), data=jobs)
                # dt['/chopped/{}'.format(folname)] = testvar  # dt['/chopped/{}'.format(folname)] = testvar

            except RuntimeError:
                #tempdata = dt.get('/chopped/{}'.format(folname))
                tempdata = dt['/chopped/{}'.format(folname)]
                #tempfol = {self.musicpath: tempdata}
                tempdata[...] = np.asarray(jobs)
        '''
        if not os.path.exists(self.f):
            with h5py.File(self.f, 'w') as dt:
                print('in chopped chopping {}'.format(folname))
                dt.create_dataset('/chopped/{}'.format(folname), data=jobs, dtype=np.float64)
                #dt['/chopped/{}'.format(folname)] = jobs  # dt['/chopped/{}'.format(folname)] = testvar
        else:
        '''

        tempfol = {folname: jobs}
        return tempfol

    def compmfcc(self, y1, sr=22050):
        S1 = l.feature.mfcc(y=y1, sr=sr, n_mfcc=40)
        return S1

    def process(self, fn):
        global raw, tempa, frr
        raw = None
        tempa = np.ones([40, 130])
        for g in fn.keys():
            bec = g
        # frr = np.ones([40,130])
        with h5py.File(self.f, 'r+') as da:
            print('/chopped/{}'.format(fn.keys()))
            #raw = np.array(da.get('/chopped/{}'.format(bec)))
            raw = da['/chopped/{}'.format(bec)][:]

        print('raw: ', raw)
        print('raw will bw printed')

        sr1 = 22050

        with Pool(self.cpu) as p:
            frr = np.dstack(p.map(self.compmfcc, raw))

        with h5py.File(self.f, 'a') as dz:
            # data = dz['/processed/{}'.format(bec)]  # load the data
            try:
                dz['/processed/{}'.format(bec)] = frr  # dz['/processed/{}'.format(fn)] = frr
            except:
                return {bec: dz['/processed/{}'.format(bec)][:]}
            dz.close()

        di = {'{}'.format(fn): frr}
        print(frr.shape)
        return (di)

    ### end process(raw)

    def lnc(self, foldername, loadfromh5py=True):
        global proc, mode

        mode = 'r' if loadfromh5py else 'r+'

        if loadfromh5py:  # if the files are processed it ge them directly from the h5py files
            with h5py.File(self.f, 'r') as dt:
                # loadfromh5py = '/processed/{}'.format(foldername) in dt

                if ('/processed/{}'.format(foldername) in dt):
                    print('in processed')
                    proc = dt['/processed/{}'.format(foldername)][:]
                    dt.close()
                    return proc
                elif ('/chopped/{}'.format(foldername) in dt):
                    print('in chopped and am processing')
                    dt.close()
                    proc = self.process(foldername)

                    return proc
                else:
                    dt.close()
                    print('from start')
                    self.lnc(foldername, loadfromh5py=False)

        else:
            breakfiles = self.loadfile(foldername)
            print(breakfiles)
            loadnchop = self.chopnload(breakfiles)
            print(loadnchop)
            proc = self.process(foldername)
            print(proc)
        return proc

    def one_hotY_genre_strings(self, sample_size = 10000):
        the_list = []
        genre_list = None
        y_one_hot = None
        total_size = 0
        with h5py.File(self.f, 'a') as dz:
            genre_list = list(dz['/processed/'].keys())
            for j, k in enumerate(genre_list):
                self.data = np.asarray(dz['/processed/' + k][:])
                if int(self.data.shape[2] ) > sample_size:
                    the_list.append(k)


            for h, i in enumerate(the_list):
                self.data = np.asarray(dz['/processed/' + i][:])

                rowno = self.data.shape[2]
                # print(h, 'data shape', data.shape, rowno)
                y_one_hot_temp = y_one_hot

                if h == 0:
                    self.Y_genre_strings = self.data
                    y_one_hot = np.ones((rowno, 1))
                    # print('{}y_one_hot:{} rowno:{}'.format(h,y_one_hot,rowno))
                    # print( h, Y_genre_strings.shape, y_one_hot.shape )
                    rowno = None
                else:
                    self.Y_genre_strings = np.dstack((self.Y_genre_strings, self.data))
                    y_one_hot = np.pad(y_one_hot, ((0, 0), (0, 1)), 'constant')
                    temp_y_one_hot = np.ones((rowno, 1))
                    temp_y_one_hot = np.pad(temp_y_one_hot, ((0, 0), (h, 0)), 'constant')
                    y_one_hot = np.vstack((y_one_hot, temp_y_one_hot))
        print(self.Y_genre_strings.shape)
        self.Y_genre_strings = np.rot90(self.Y_genre_strings, k=1, axes=(2, 0))
        print(self.Y_genre_strings.shape)
        #self.Y_genre_strings = np.rot90(self.Y_genre_strings, k=1, axes=(1, 2))
        #print(self.Y_genre_strings.shape)
        # print(Y_genre_strings.shape,  y_one_hot.shape)
        '''
        y_one_hot = np.zeros((Y_genre_strings.shape[0], len(genre_list)))
        for i, genre_string in enumerate(Y_genre_strings):
            index = genre_list.index(genre_string)
            y_one_hot[i, index] = 1
        '''
        arr = np.arange(len(y_one_hot))
        np.random.shuffle(arr)
        print('will start writing')
        print(y_one_hot.shape, self.Y_genre_strings.shape)
        #shuff_Y_genre_strings = Y_genre_strings[arr]
        #shuff_y_one_hot = y_one_hot[arr]
        #print(shuff_Y_genre_strings.shape, shuff_y_one_hot.shape)
        with h5py.File(self.p, 'w') as svr:
            svr['mat_stack'] = self.Y_genre_strings
            svr['one_stack'] = y_one_hot
            svr['shuff_mat'] = self.Y_genre_strings[arr]
            svr['shuff_one'] = y_one_hot[arr]

           #svr['genre_list'] = genre_list
        return None

    def main(self):

        '''
        dp.merge_audio_files()
        dp.sorte()
        '''

        genre_rec = []
        # if exists('C:/Users/sdave/data/sorted/genre_list.h5py'):
        #    with h5py.File('C:/Users/sdave/data/sorted/genre_list.h5py', 'r') as da:
        #        genre_rec = da['genre_record'][:]
        # "/raid/dataset/sorted/"

        if exists(self.g):
            genre_rec = pickle.load(open(self.g, "rb"))
        print(listdir(self.sorted_audio_path))
        for i in listdir(self.sorted_audio_path):

            print(i)
            if isdir(self.sorted_audio_path + i) and len(listdir(self.sorted_audio_path + i)) > 10:
                if i in genre_rec[:-1]:
                    continue
                else:
                    try:
                        if i != genre_rec[-1]:
                            genre_rec.append(i)
                    except IndexError:
                        genre_rec.append(i)
                    # with h5py.File('C:/Users/sdave/data/sorted/genre_list.h5py', 'r+') as da:
                    # b.sorte()
                    print('loadfile')
                    testloadfile = self.loadfile(i)
                    print('in chop file')
                    testloadnchop = self.chopnload(testloadfile)
                    print('in process')
                    testprocess = self.process(testloadnchop)

                    # with h5py.File('C:/Users/sdave/data/sorted/genre_list.h5py', 'w') as ta:
                    #    ta['genre_record'] = genre_rec
                    # Save a dictionary into a pickle file.
                    print(genre_rec)
                    if exists(self.g):
                        os.remove(self.g)

                    pickle.dump(genre_rec, open(self.g, "w+b"))  # Y_genre_strings, y_one_hot, genre_list = dp.one_hotY_genre_strings()

        #self.one_hotY_genre_strings()
    '''
        import fma
        tracks = fma.load('data/fma_metadata/tracks.csv')
        subset = tracks.index[tracks['set', 'subset'] <= 'medium']
        labels = tracks.loc[subset, ('track', 'genre_top')]
        labels.name = 'genre'
        labels.to_csv('data/train_labels.csv', header=True)    
    '''


if __name__ == '__main__':
    b_obj = DataProcess()
    #b_obj.main()
    b_obj.one_hotY_genre_strings()
