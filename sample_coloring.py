#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from colour import Color
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
import statistics as st
import time
import random
import re
from random import randrange

from granatum_sdk import Granatum

COLORS = ["#3891ea", "#29ad19", "#ac2d58", "#db7580", "#ed2310", "#ca2dc2", "#5f7575", "#7cc1b5", "#c3bd78", "#4ffa24"]

def resetArray(minX, maxX, numsteps, numsigfigs):
    powten = 10**numsigfigs
    minX = minX*powten
    maxX = maxX*powten
    newMinX = np.floor(minX)
    newMaxX = np.ceil(maxX)
    stepsize = np.ceil((newMaxX-newMinX)/(numsteps-1.0))
    newMaxX = newMinX+stepsize*(numsteps-1.0)
    midNew = (newMaxX+newMinX)/2.0
    midOld = (maxX+minX)/2.0
    shift = np.round(midOld-midNew)
    return np.arange(newMinX+shift, newMaxX+stepsize+shift, step=stepsize)/powten

def main():
    tic = time.perf_counter()

    gn = Granatum()
    sample_coords = gn.get_import("viz_data")
    value = gn.get_import("value")
    coloring_type = gn.get_arg("coloring_type")
    bounding_stdev = gn.get_arg("bounding_stdev")
    label_location = gn.get_arg("label_location")
    label_transform = gn.get_arg("label_transform")
    labelXaxis = gn.get_arg("labelXaxis")
    labelYaxis = gn.get_arg("labelYaxis")
    sigfigs = gn.get_arg("sigfigs")
    numticks = gn.get_arg("numticks")
    font = gn.get_arg('font')

    coords = sample_coords.get("coords")
    dim_names = sample_coords.get("dimNames")
    seed = gn.get_arg('random_seed')
    random.seed(seed)
    np.random.seed(seed)

    df = pd.DataFrame(
        {"x": [a[0] for a in coords.values()], "y": [a[1] for a in coords.values()], "value": pd.Series(value)},
        index=coords.keys()
    )

    target_dpi=300
    target_width=7.5 # inches
    target_height=6.5 # inches
    font_size_in_in=font/72.0 # inches
    font_size_in_px=font_size_in_in*target_dpi

    try:

        if coloring_type == "categorical":
            uniq = df["value"].unique()
            uniq.sort(kind="stable")
            num = uniq.shape[0]
            COLORS2 = plt.get_cmap('gist_rainbow')
            carr = [0]*df.shape[0]
            listcats = list(df["value"])     
            miny = min(list(df["y"]))
            maxy = max(list(df["y"]))
            scaley = (maxy-miny)/(target_height*target_dpi)
            print("Scaley = {}".format(scaley))
            colorhash = {}
            colorstep = np.ceil(256.0/num)
            coffset = randrange(colorstep)
            grouptocolor = np.random.choice(np.arange(num), num, replace=False)

            legend_handles = []

            for i, cat in enumerate(uniq):
                dff = df[df["value"] == cat]
                if dff.shape[1] < 1:
                    continue
                xs = list(dff["x"])
                ys = list(dff["y"])
                #avgx = sum(dff["x"]) / len(dff["x"]) 
                #avgy = sum(dff["y"]) / len(dff["y"]) 
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=COLORS[i].hex_l, label=cat)
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=[abs(hash(cat)) % 256]*len(dff["x"]), cmap=COLORS2, label=cat)
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=abs(hash(cat)) % 256, cmap=COLORS2, label=cat)
                #abs(hash(cat))
                colorindex = (coffset + grouptocolor[i]*colorstep) % 256
                colorhash[cat] = colorindex
                craw = COLORS2((colorindex+0.0)/256.0)
                clr = [craw[0], craw[1], craw[2], 0.2]
                whitetransparent = [1.0, 1.0, 1.0, 0.5]
                coloropaque = [craw[0], craw[1], craw[2], 1.0]
                if len(xs)>3:
                    pts = list(zip(xs, ys))
                    cent = np.mean(pts, axis=0)
                    lengs = list(map(lambda p: math.sqrt((p[0]-cent[0])*(p[0]-cent[0])+(p[1]-cent[1])*(p[1]-cent[1])), pts))
                    avgleng = st.mean(lengs)
                    stdleng = st.stdev(lengs)*bounding_stdev
                    rpts = []
                    if(stdleng > 0.0):
                        for j, ln in enumerate(lengs):
                            if(ln - avgleng < stdleng):
                                rpts.append(pts[j])
                        pts = rpts
                    cent = np.mean(pts, axis=0)
                    hull = ConvexHull(pts)
                    ptslist = []
                    for pt in hull.simplices:
                        ptslist.append(pts[pt[0]])
                        ptslist.append(pts[pt[1]])
                    ptslist.sort(key=lambda p: np.arctan2(p[1]-cent[1], p[0]-cent[0]))
                    ptslist = ptslist[0::2]
                    ptslist.insert(len(ptslist), ptslist[0])
                    lowestpt = ptslist[0]
                    if label_location == 'bottom':
                        for pt in ptslist:
                            if(pt[1] < lowestpt[1]):
                                lowestpt = pt
                    else:
                        lowestpt = ptslist[randrange(len(ptslist))]
                    if(bounding_stdev >= 0.0):
                        poly = Polygon(1.1*(np.array(ptslist)-cent)+cent, facecolor=clr)
                        poly.set_capstyle('round')
                        plt.gca().add_patch(poly)
                        poly.set_color(clr)
                    label_text = cat
                    if label_transform == "numbers":
                        label_text = re.sub("[^0-9]", "", cat)
                    if label_location == 'legend':
                        patch = mpatches.Patch(color=coloropaque, label=label_text)
                        legend_handles.append(patch)
                    else:
                        txt = plt.text(lowestpt[0], lowestpt[1]-scaley*font_size_in_px*1.2, label_text, fontsize=font, fontname="Arial", ha="center", va="center", color="black", bbox=dict(boxstyle="round",fc=whitetransparent,ec=coloropaque))
                    # plt.gca().add_artist(txt)
                for j,x in enumerate(listcats):
                    if x == cat:
                        carr[j] = colorhash[cat]
                        #carr[j] = colorhash[cat] / 256.0
                        #int(abs(hash(cat)) % 256)
            
            plt.scatter(x=df["x"], y=df["y"], s=5000 / df.shape[0], c=carr, cmap=COLORS2)
            #lgd = plt.legend(markerscale=6, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
            if label_location == "legend":
                plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        elif coloring_type == "continuous":
            plt.scatter(x=df["x"], y=df["y"], s=5000 / df.shape[0], c=df["value"], cmap="Reds")
            plt.colorbar()

        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        # stepsizex=(xmax-xmin)/numticks
        # stepsizey=(ymax-ymin)/numticks
        xtickArray = resetArray(xmin, xmax, numticks, sigfigs)
        ytickArray = resetArray(ymin, ymax, numticks, sigfigs)
        # plt.xticks(np.arange(xmin, xmax+stepsizex, step=stepsizex), fontsize=font, fontname="Arial")
        # plt.yticks(np.arange(ymin, ymax+stepsizey, step=stepsizey), fontsize=font, fontname="Arial")
        plt.xlim(xtickArray[0], xtickArray[-1])
        plt.ylim(ytickArray[0], ytickArray[-1])
        plt.xticks(xtickArray, fontsize=font, fontname="Arial")
        plt.yticks(ytickArray, fontsize=font, fontname="Arial")
        if labelXaxis == "":
            plt.xlabel(dim_names[0], fontsize=font, fontname="Arial")
        else:
            plt.xlabel(labelXaxis, fontsize=font, fontname="Arial")

        if labelYaxis == "":
            plt.ylabel(dim_names[1], fontsize=font, fontname="Arial")
        else:
            plt.ylabel(labelYaxis, fontsize=font, fontname="Arial")

        # plt.tight_layout()

        gn.add_current_figure_to_results(
            "Scatter-plot",
            dpi=target_dpi,
            width=target_width*target_dpi,
            height=target_height*target_dpi,
            savefig_kwargs={'bbox_inches': 'tight'}
        )

        toc = time.perf_counter()
        time_passed = round(toc - tic, 2)

        timing = "* Finished sample coloring step in {} seconds*".format(time_passed)
        gn.add_result(timing, "markdown")

        gn.commit()


    except Exception as e:

        plt.figure()
        plt.text(0.05, 0.7, 'Values used as colors and type of sample metadata are incompatible with each other')

        if coloring_type == 'categorical':
            new_coloring_type = 'continuous'
        else:
            new_coloring_type = 'categorical'


        plt.text(0.05, 0.5, 'Retry the step with ' + new_coloring_type + ' instead of ' + coloring_type) 
        plt.axis('off')
        gn.add_current_figure_to_results('Scatter-plot')

        gn.commit()



if __name__ == "__main__":
    main()
