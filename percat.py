import re
import json
from tqdm import tqdm
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import brokenaxes

from pycocotools.coco import COCO
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

mode = "linear"     # choose from ["linear", "log", "broken"]. Better not use "log"
style = "hline"    # choose from ["box", "hline"]

id_to_py={}
py_to_eng={}

def load_name(jd):
    for cate in jd['categories']:
        id_to_py[cate['id']] = cate['name']
    with open('fooddet100k_eng.txt',encoding='utf-8') as f:
        for i in f.readlines():
            tt = i.strip('\n').split('\t')
            if tt[1] not in py_to_eng:
                py_to_eng[tt[1]] = tt[2]

def draw(tiktok:int, numlist1: list, xindex_list1: list, name_list):
    width1 = 0.9
    index1 = list(range(0,len(numlist1),tiktok))    # 刻度
    index2 = range(len(numlist1))             # 统计图柱
    tick_spacing = 10

    # index1.pop()

    if mode in ["linear", "log"]:       # 线性轴或对数轴
        plt.ylim(0, 1)
        fig, ax1 = plt.subplots(1,1, figsize=(44, 16))  # 1幅子图（1行1列）
        ax1.margins(0.01, 0.02)

        # 添加纵横轴的刻度
        # log, 截断, 还是原来的？还有什么选项？
        # ax1.set_xlabel("ID of food categories",fontsize=50)   # or index?
        ax1.set_ylim(0,1.1)
        ax1.set_ylabel("mAP per food category",fontsize=50)
        ax1.set_xticks(index1)
        ax1.set_xticklabels(name_list,rotation = 45,fontsize=35,ha='right')
        y_ticks = list(np.arange(0, 1.1, 0.1))
        ax1.set_yticks(y_ticks)
        ylabels = ['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
        ax1.set_yticklabels(ylabels,fontsize=35)
        if mode == "log":
            ax1.set_yscale("log")
        ax1.bar(index2, numlist1, width = width1, color="red", edgecolor="#000000", linewidth=1)
        if style == "hline":
            ax1.spines["left"].set_visible(False)
            ax1.spines["right"].set_visible(False)
            ax1.spines["top"].set_visible(False)
            ax1.grid(axis="y")
            ax1.tick_params(axis='y', which='both', length=0)
            ax1.set_axisbelow(True)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        plt.tight_layout()


    elif mode in ["broken"]:        # 截断坐标轴
        fig, axes = plt.subplots(2, 1, figsize=(80, 34))
        y_lims = [[33000, 34501, 1500], [0, 11000, 1000]]
        coord = {"left": 0.07, "bottom": 0.12, "width": 0.9, "height_b": 0.70, "gap": 0.02, "height_t": 0.08}
        for i, ax in enumerate(axes):
            ax.margins(0.01, 0.02)
            ax.bar(index2, numlist1, width = width1, label="num", color="#0ECEFA", edgecolor="#000000", linewidth=1)
            ax.set_ylim(y_lims[i][0], y_lims[i][1])
            y_ticks = list(np.arange(y_lims[i][0], y_lims[i][1], y_lims[i][2]))
            if i == 1 and y_ticks[-1] != 11000:
                y_ticks.append(11000)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_ticks, fontsize=60)

            if style == "hline":
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(axis="y")
                ax.tick_params(axis='y', which='both', length=0)
                ax.set_axisbelow(True)

        axes[0].tick_params(labeltop=False, labelbottom=False, top=False, bottom=False)
        axes[1].set_xticks(index1)
        axes[1].set_xticklabels(name_list, rotation = 35, fontsize=60,ha='right')
        axes[1].set_position([coord["left"], coord["bottom"], coord["width"], coord["height_b"]])
        axes[0].set_position([coord["left"], coord["bottom"]+coord["height_b"]+coord["gap"], coord["width"], coord["height_t"]])
        #axes[1].set_xlabel("Id of food categories",fontsize=45)   # or index?
        axes[1].set_ylabel("mAP of food categories",fontsize=60)

    fig.savefig('stat_BpC1.png',dpi=300)
    # print(numlist1[:10])


def show_catap(apcat, jsonname):

    with open(jsonname, 'r', encoding='utf-8')as fp:
        json_data = json.load(fp)

    load_name(json_data)

    stat_IpC = {}

    for bbox in json_data['annotations']:
        # index
        key = bbox['category_id']
        # name
        # key_list = [x['name'] for x in json_data['categories'] if x['id'] == bbox['category_id']]
        # key = len(key_list) > 0 ? key_list[0] : "UNKNOWN"
        if key not in stat_IpC:
            stat_IpC[key] = 1
        else:
            stat_IpC[key] = stat_IpC[key] + 1

    # 依照 cate_id 升序排列
    xindex_list0 = []
    numlist0=[]
    for k in sorted(stat_IpC):
        numlist0.append(stat_IpC[k])
        xindex_list0.append((k))

    stat_IpCs = sorted(stat_IpC.items(), key=lambda stat_IpC: stat_IpC[1], reverse=True)

    # 依照 标注框数量 降序排列
    numlist1 = []
    for i in stat_IpCs:
        numlist1.append(i[1])

    tiktok=10    # 在统计图中每20个柱绘制一个刻度，这样较美观

    xindex_list1 = []
    name_list = []
    tc=-1
    for tt in range(len(stat_IpCs)):
        i = stat_IpCs[tt]
        tc = tc + 1
        if tc % tiktok == 0:
            xindex_list1.append(i[0])
            temp_name=py_to_eng[id_to_py[i[0]]]
            if len(temp_name) > 14:
                fl=False
                for cc in range(0,100):
                    tt-=1
                    i = stat_IpCs[tt]
                    temp_name=py_to_eng[id_to_py[i[0]]]
                    if len(temp_name) <= 13:
                        fl=True
                        break
                tt=tt+100
                if not fl:
                    for cc in range(0,100):
                        tt+=1
                        i = stat_IpCs[tt]
                        temp_name=py_to_eng[id_to_py[i[0]]]
                        if len(temp_name) <= 12:
                            break
            print(temp_name)
            name_list.append(temp_name)

    print("Min_id:",stat_IpCs[numlist1.index(min(numlist1))])
    print("Max_id:",stat_IpCs[numlist1.index(max(numlist1))])

    numlist2 = []
    for i in apcat:
        numlist2.append(float(i[1]))

    draw(tiktok, numlist2, xindex_list1, name_list)

    def by_score(t): #down to up
        return t[1]

    ap2 = sorted(apcat, key=by_score)

    print(ap2)

    print("Done")
