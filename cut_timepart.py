import numpy as np
from pyedflib import highlevel
import os
import datetime



def scandir(dir_path, ext=None):
    """
    Returns the list of included files
    two ways of use (example):
    scandir(path,'.edf') - finds all files with .edf extension
    scandir(path,['.edf', 'txt']) - finds all files with .edf and .txt extension
    """
    def files_list(all_files, extension):
        res = []
        for file in all_files:
            if file.endswith(extension):
                res.append(file)
        return res
    listed_dir = os.listdir(dir_path)
    if type(ext) is str:
        listed_dir = files_list(listed_dir, ext)
    elif type(ext) is list:
        res2 = []
        for inst in ext:
            res2.append(np.array(files_list(listed_dir, inst)))
        to_concat_lists = tuple(res2)
        listed_dir = np.concatenate(to_concat_lists).tolist()
    return listed_dir

def create_floder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_edf(edfpath="./edf_file.edf", digital=False):
    signals, signal_headers, header = highlevel.read_edf(edfpath, digital=digital)

    #-----Uncoment if wrong fs-------------------------
    # manual_fs_mass = [200, 200, 200, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1]
    # for header,manual_fs in zip(signal_headers, manual_fs_mass):
    #     header['sample_rate'] =manual_fs
    #     header['sample_frequency'] = manual_fs
    #---------------------------------------------------

    return signals, signal_headers, header


def write_fast_edf(signalss, signal_headerss, headerr, pp='./tmp.edf', ):
    highlevel.write_edf(pp, signalss, signal_headerss, headerr, digital=False, file_type=-1)


def split_edf(input_file, out_dir_path, fromTimeSec, durationSec):
    def get_fs_mas(headers):
        fs_mass = []
        for header in headers:
            fs = (header['sample_rate'])
            fs_mass.append(fs)
        return fs_mass

    def get_shortest_time_cut(signal_mass, fs_massive, duration_sec):
        print('fs_mas = ' + str(fs_massive))
        timecut_mass = []
        for sig, fs in zip(signal_mass,fs_massive):
            print(fs)
            dur_samples = (duration_sec*fs).__floor__()
            total_timecut = (dur_samples/fs).__floor__()
            timecut_mass.append(total_timecut)
        return min(timecut_mass)

    def calc_stopcount(starting_count, mass_of_fs, shortesst_time_cut):
        stopping_count = []
        for st_c, the_fs, in zip(starting_count, mass_of_fs):
            new_count = st_c + round(shortesst_time_cut* the_fs)
            stopping_count.append(new_count)
        return stopping_count

    def cut_signals(sigmass, start_countt, stop_countt):
        new_sigmass = []
        for signal, start_c, stop_c in zip(sigmass, start_countt, stop_countt):
            new_signal = signal[start_c:stop_c]
            new_sigmass.append(new_signal)
        return new_sigmass
        # return np.array(new_sigmass)

    print('loading file '+ input_file)
    signals, signal_headers, header = load_edf(edfpath=input_file)

    fs_mas = get_fs_mas(signal_headers)
    # raw_data, info, channels = load_edf2(edfpath=ffp)
    # fs = float(info['sfreq'])

    shortest_time_cut = get_shortest_time_cut(signals,fs_mas, durationSec)
    shortest_time_cut_from_sample = get_shortest_time_cut(signals,fs_mas,fromTimeSec)
    print('shortest time cut: ' + str(shortest_time_cut))
    file_name = input_file.split('/')[-1].replace('.edf','').replace('.EDF','')

    zero_pos = list(np.zeros([1, len(signals)]).astype(int)[0])
    start_count = calc_stopcount(zero_pos,fs_mas,shortest_time_cut_from_sample)

    stop_count = calc_stopcount(start_count,fs_mas, shortest_time_cut)
    cutted_part = cut_signals(signals,start_count,stop_count)

    total_signal_time = len(cutted_part[0]) / fs_mas[0]
    for part, fs in zip(cutted_part, fs_mas):
        persignal_time = len(part) / fs
        print('per_signal_time '+ str(persignal_time))
        if persignal_time == total_signal_time:
            total_signal_time = persignal_time
        else:
            # pass
            raise Exception('Signals not equal, got header error!, update the code to solve')

    outfpath = out_dir_path+file_name+'_from'+ str(fromTimeSec)+'.edf'
    print('Storing:' + str(outfpath))
    write_fast_edf(cutted_part,signal_headers,header,pp=outfpath)

input_edf_path = 'input_edf/tmp/'
results_path = 'result_splited_edfs/tmp/'

def calc_from_time_sec(input_file_path,y,m,d,h,min,s):
    signals, signal_headers, header = load_edf(input_file_path)
    startDate = header['startdate']
    # Create a datetime object
    target_date = datetime.datetime(y, m, d, h, min, s)
    time_delta = (target_date - startDate).total_seconds()
    return time_delta

from_time_sec = 0
dur_sec = 60*60#


files_list = scandir(input_edf_path, ['.edf', '.EDF'])
for file in files_list:
    # out_file_dir_path = results_path + file.replace('.edf', '').replace('.EDF', '') + '/'
    # create_floder(out_file_dir_path)

    out_file_dir_path = results_path

    input_file_path = input_edf_path + file
    split_edf(input_file_path, out_file_dir_path, from_time_sec, dur_sec)

print('Check the output in edf viewer !!!!!!')




