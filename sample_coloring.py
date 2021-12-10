#!/usr/bin/env python

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from colour import Color
from matplotlib.patches import Polygon
import statistics as st
import time
import random
from random import randrange

from granatum_sdk import Granatum

COLORS = ["#3891ea", "#29ad19", "#ac2d58", "#db7580", "#ed2310", "#ca2dc2", "#5f7575", "#7cc1b5", "#c3bd78", "#4ffa24"]

def main():
    tic = time.perf_counter()

    gn = Granatum()
    sample_coords = gn.get_import("viz_data")
    value = gn.get_import("value")
    coloring_type = gn.get_arg("coloring_type")
    bounding_stdev = gn.get_arg("bounding_stdev")
    font = gn.get_arg('font')

    coords = sample_coords.get("coords")
    dim_names = sample_coords.get("dimNames")
    random.seed(gn.get_arg('random_seed'))

    df = pd.DataFrame(
        {"x": [a[0] for a in coords.values()], "y": [a[1] for a in coords.values()], "value": pd.Series(value)},
        index=coords.keys()
    )

    try:

        if coloring_type == "categorical":
            uniq = df["value"].unique();
            num = uniq.shape[0]
            COLORS2 = plt.get_cmap('gist_rainbow')
            carr = [0]*df.shape[0]
            listcats = list(df["value"])     
            miny = min(list(df["y"]))
            maxy = max(list(df["y"]))
            scaley = (maxy-miny)/650
            print("Scaley = {}".format(scaley))
            colorhash = {}
            colorstep = np.ceil(256/num)
            coffset = randrange(colorstep)
            grouptocolor = np.random.choice(np.arange(num), num, replace=False)

            for i, cat in enumerate(uniq):
                dff = df[df["value"] == cat]
                xs = list(dff["x"])
                ys = list(dff["y"])
                #avgx = sum(dff["x"]) / len(dff["x"]) 
                #avgy = sum(dff["y"]) / len(dff["y"]) 
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=COLORS[i].hex_l, label=cat)
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=[abs(hash(cat)) % 256]*len(dff["x"]), cmap=COLORS2, label=cat)
                #plt.scatter(x=dff["x"], y=dff["y"], s=5000 / df.shape[0], c=abs(hash(cat)) % 256, cmap=COLORS2, label=cat)
                #abs(hash(cat))
                colorindex = (coffset + grouptocolor[i]) % 255
                colorhash[cat] = colorindex
                craw = COLORS2(colorindex/255.0)
                color = (craw[0], craw[1], craw[2], 0.2)
                whitetransparent = (1, 1, 1, 0.5)
                coloropaque = (craw[0], craw[1], craw[2], 1.0)
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
                    for pt in ptslist:
                        if(pt[1] < lowestpt[1]):
                            lowestpt = pt
                    if(bounding_stdev >= 0.0):
                        poly = Polygon(1.1*(np.array(ptslist)-cent)+cent, facecolor=color)
                        poly.set_capstyle('round')
                        plt.gca().add_patch(poly)
                    plt.text(lowestpt[0], lowestpt[1]-scaley*10, cat, fontsize=font, fontname="Arial", ha="center", va="center", color="black", bbox=dict(boxstyle="round",fc=whitetransparent,ec=coloropaque))
                for j,x in enumerate(listcats):
                    if x == cat:
                        carr[j] = colorhash[cat] / 256.0
                        #int(abs(hash(cat)) % 256)
            
            plt.scatter(x=df["x"], y=df["y"], s=5000 / df.shape[0], c=carr, cmap=COLORS2)
            lgd = plt.legend(markerscale=6, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    #60 / (5000 / df.shape[0])
        elif coloring_type == "continuous":
            plt.scatter(x=df["x"], y=df["y"], s=5000 / df.shape[0], c=df["value"], cmap="Reds")
            plt.colorbar()

        xmin, xmax = plt.gca().get_xlim()
        ymin, ymax = plt.gca().get_ylim()
        stepsizex=(xmax-xmin)/5.0
        stepsizey=(ymax-ymin)/5.0
        plt.xticks(np.arange(xmin, xmax+stepsizex, step=stepsizex), fontsize=font, fontname="Arial")
        plt.yticks(np.arange(ymin, ymax+stepsizey, step=stepsizey), fontsize=font, fontname="Arial")
        plt.xlabel(dim_names[0], fontsize=font, fontname="Arial")
        plt.ylabel(dim_names[1], fontsize=font, fontname="Arial")
        # plt.tight_layout()

        target_dpi=300
        target_width=7.5 # inches
        target_height=6.5 # inches

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
