import os
import urllib.request
import pydub
import argparse

'''
使用手册
传入需要输出成音频的单词.txt文件,与该程序在同一路径,便可以生成想要的音频.
需要一段一秒静音的音频,已经在例子中给出. (one_second.mp3)
'''

DEBUG_MODE = False

parser = argparse.ArgumentParser(description="FzSoundwords")
parser.add_argument('--selectvoice', '-s', help='选择读音偏好(0:美音  1:英音  2:全部) || Choose a pronunciation preference.', type=int, default=2)
parser.add_argument('--timeinterval', '-t', help='单词间间隔时间(秒) || The time between words(s).', type=int, default=1)
args = parser.parse_args()
SELECT_VOICE = args.selectvoice  # 0:美音  1:英音  2:全部
TIME_INTERVAL = args.timeinterval  # 每个单词间间隔 单位:秒

def print_logo():
    print("""
 #######################################################
##                                                     ##
##       FFFFFFFFFFF                                   ##
##       FFFFFFFFFF                                    ##
##       FFF                                           ##
##       FFF                                           ##
##       FFFFFFF       zzzzzzzzzz    zzzzzzzzzz        ##
##       FFFFFF              zz            zz          ##
##       FFF               zz            zz            ##
##       FFF             zz            zz              ##
##       FF            zzzzzzzzzz    zzzzzzzzzz        ##
##                                                     ##
##    (^_^)#                                           ##
##      Author:Feng Zhengzhan   Project:FzSoundwords   ##
##                                                     ##
 #######################################################
    """)
    print(" (^_^)# 欢迎使用烽征战-单词文件合成音频项目")
    print(" (^_^)# 项目耗时取决于单词数量，给程序点耐心哟 ")

class youdao():
    def __init__(self, filename):
        '''
        调用youdao API
        type = 0：美音
        type = 1：英音
        http://dict.youdao.com/dictvoice?type=0&audio=    %20 美音
        http://dict.youdao.com/dictvoice?type=1&audio=    %20 英音
        '''

        self.dirRoot = os.path.dirname(os.path.abspath(__file__))  # 当前文件根目录
        self.filename = filename.lower()  # 文件名称
        self.filename_txt = str(self.filename) + ".txt"  # 文件总名称
        self.filename_tmp = self.filename + "_tmp"
        self.filename_allpath = os.path.join(self.dirRoot, self.filename_txt)

        if DEBUG_MODE:
            print("[1]变量值")
            print("self.filename", self.filename)
            print("self.filename_txt", self.filename_txt)
            print("self.filename_tmp", self.filename_tmp)
            print("self.dirRoot", self.dirRoot)
            print("self.dirRoot", self.dirRoot)

        # 分片目录
        self.one_list = []
        self.all_list = []

        self.url_path = ""
        self.name = ["_US", "_EN"]  # 美音、英音后缀

        if not os.path.exists(self.filename_tmp):  # 创建单词临时目录
            os.makedirs(self.filename_tmp)

        # 加载单词列表
        self.all_word = []
        with open(self.filename_allpath, "r", encoding="utf-8") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                # 将空行和多余字符去除
                words_split = line.split(" ")
                word = ""
                for one in words_split:
                    if one in ['', '\n']:
                        continue
                    one = one.replace("\n", "").replace("\r", "")
                    if one.isalpha():
                        word += one + "%20"
                    else:
                        break
                if word not in ['', '\n']:
                    self.all_word.append(word[:-3])

        if DEBUG_MODE:
            print("[2]单词列表", self.all_word)


    def down(self):
        # 通过有道词典接口，下载单词的MP3
        try:
            for i in range(len(self.all_word)):
                if SELECT_VOICE == 2:
                    for j in range(0, 2):
                        self.url_path = "http://dict.youdao.com/dictvoice?type=" + str(j) + "&audio=" + str(self.all_word[i])
                        self.one_list.append(str(self.all_word[i]) + str(self.name[j]) + ".mp3")
                        tmp_filename = str(self.filename_tmp) + os.sep + str(self.all_word[i]) + str(self.name[j]) + ".mp3"
                        urllib.request.urlretrieve(self.url_path, filename=tmp_filename)
                        print("[+] Finish {0} : {1}!".format(tmp_filename, self.url_path))
                elif SELECT_VOICE == 0 or SELECT_VOICE == 1:
                    self.url_path = "http://dict.youdao.com/dictvoice?type=" + str(SELECT_VOICE) + "&audio=" + str(self.all_word[i])
                    self.one_list.append(str(self.all_word[i]) + str(self.name[SELECT_VOICE]) + ".mp3")
                    tmp_filename = str(self.filename_tmp) + os.sep + str(self.all_word[i]) + str(self.name[SELECT_VOICE]) + ".mp3"
                    urllib.request.urlretrieve(self.url_path, filename=tmp_filename)
                    print("[+] Finish {0} : {1}!".format(tmp_filename, self.url_path))
                self.all_list.append(self.one_list)
                self.one_list = []
        except Exception as e:
            print(str(self.url_path) + "Exception :" + str(e))


    def merge(self):
        # 将音频合并输出
        try:
            print("[-] Writting.", end="")
            song_one = pydub.AudioSegment.from_mp3(self.dirRoot + os.sep + 'one_second.mp3')
            song_all = pydub.AudioSegment.from_mp3(self.dirRoot + os.sep + 'one_second.mp3')
            for i in range(len(self.all_list)):
                for j in range(len(self.all_list[i])):
                    # pydub.AudioSegment.converter = "D:\\ffmpeg\\bin"
                    tmp_filename = self.dirRoot + os.sep + str(self.filename_tmp) + os.sep + str(self.all_list[i][j])
                    pydub.AudioSegment.from_file(tmp_filename).export(tmp_filename, format='mp3')
                    song_all += pydub.AudioSegment.from_mp3(tmp_filename)
                for time in range(0, TIME_INTERVAL):  # 不同单词间隔时间
                    song_all += song_one
                print(".", end="")
            # 简单输入合并之后的音频
            song_all.export(self.dirRoot + os.sep + str(self.filename) + "_soundwords.mp3", format='mp3')
            print("[+] Finish merge !")
        except Exception as e:
            print("词根:" + str(self.filename) + "Exception :" + str(e) )


    def delete_tmp(self):  # 删除临时文件
        if not DEBUG_MODE:
            for i in range(len(self.all_list)):
                for j in range(len(self.all_list[i])):
                    tmp_filename = self.dirRoot + os.sep + str(self.filename_tmp) + os.sep + str(self.all_list[i][j])
                    if os.path.exists(tmp_filename):
                        os.remove(tmp_filename)
            os.rmdir(self.dirRoot + os.sep + str(self.filename_tmp))



if __name__ == "__main__":
    print_logo()
    while True:
        try:
            TXT_FILENAME = input("请输入.txt文件的文件名xxx即可(输入qqq可退出) || Input filename (qqq exit):")
            if TXT_FILENAME == "qqq":
                break
            TXT_FILENAME = TXT_FILENAME.strip()
            voice = youdao(TXT_FILENAME)
            voice.down()
            voice.merge()
            voice.delete_tmp()
        except Exception as e:
            print("[*] error: {0}.".format(e))
