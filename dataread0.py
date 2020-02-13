import csv
####################EARTH DISTANCE FINDER#####################
from math import radians, sin, cos, asin, sqrt
def distfind(src,dest):
    slon,slat = src[0],src[1]
    elon,elat = dest[0],dest[1]
    slat=radians(float(slat))
    slon=radians(float(slon))
    elat=radians(float(elat))
    elon=radians(float(elon))
    dlat=elat-slat
    dlon=elon-slon
    a = sin(dlat/2)**2 + cos(slat)*cos(elat) * sin(dlon/2)**2
    #c = 2 * atan2(sqrt(a), sqrt(1-a))
    dist=2*6371010*asin(sqrt(a))
    return dist
    #API based
    #gmaps = googlemaps.Client(key='AIzaSyANek40q60bomgvZAYTs5sOSrMZTh9AFn8')
    #print(gmaps.distance_matrix((40.714224, -73.961452),(116.43003,39.93209))['rows'][0]['elements'][0]['distance']['value'])
    
from sklearn.metrics import pairwise_distances
def diff_affinity(X):
    return pairwise_distances(X, metric=distfind)

#######################GET TIME INTERVAL (b/w consecutive readings)####################
from datetime import datetime
def timeint(start, end):
    sdt = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    edt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    timel=(edt-sdt).total_seconds()
    return timel

###############READ TSMC DATASET WITH TIMESTAMP ORDERING IN CSV####################
import os
path="C:/Users/mayank/Downloads/ANIMESH/datasets/dataset_tsmc2014/"
trlist=[]
for file in os.scandir(path):
    with open(file, 'r') as c_file:
        cread=csv.reader(c_file,delimiter='\t')
        lines=0
        for row in cread:
            stx=row[7]
            st=stx.replace('+0000 ','')
            dt=datetime.strptime(st,'%a %b %d %H:%M:%S %Y')
            #coord=row[4].split('\t')
            #print(coord)
            trlist.append([float(row[4]),float(row[5]),dt])

########################AGGLOMERATIVE CLUSTERING FOR ALL POINTS###############
from sklearn.cluster import AgglomerativeClustering
clusters = AgglomerativeClustering(n_clusters=None, affinity=diff_affinity, compute_full_tree='auto', connectivity=None, distance_threshold=100.0, linkage='complete')
clusters.fit(trlist[:2]) 
y=clusters.labels_

########################CLUSTERING BY VARIANCE IN DATA########################
def bir(a,k):
    wav=(k*np.var(a[:k]) + (len(l)-k)*np.var(a[k:]))/len(l)
    if np.var(a)-wav:
        return k
    else:
        return (l,k+1)

#######################READ GEOLIFE DATASET###################################
from pathlib import Path
import numpy as np

path3="C:/Users/Animesh/Documents/vta research project/DATASETS/Geolife Trajectories 1.3/Data/"
entry=Path(path3)
ctr=0
trajectories=[]
coords=[]
for direc in entry.iterdir():
    #print(path3+direc.name)
    for x in os.scandir(path3+direc.name+"/Trajectory/"):
        trlist=[]
        with open(x, mode='r') as dtfile:
            rd=csv.reader(dtfile,delimiter=',')
            ct=0
            for row in rd:
                ct+=1
                if ct<7:
                    continue
                trlist.append([row[0],row[1],row[4]])
                coords.append([float(row[0]),float(row[1])])
        if len(trlist)>3:
            trajectories.append(trlist)
#Range of the dataset
bounds=[[min(coords[:,0]),min(coords[:,1])],[max(coords[:,0]),max(coords[:,1])]]

#################TIME DISTRIBUTION IN GEOLIFE DATASET##################
times=[]
for direc in entry.iterdir():
    for x in os.scandir(path3+direc.name+"/Trajectory/"):
        with open(x, mode='r') as dtfile:
            rd=csv.reader(dtfile,delimiter=',')
            ct=0
            for row in rd:
                ct+=1
                if ct<7:
                    continue
                times.append(int(row[6][:2]))
                
time_ranges = np.zeros(24, dtype='uint8')
for t in times:
    time_ranges[t]+=1

import matplotlib.pyplot as plt

y_pos = np.arange(24)
plt.bar(y_pos, time_ranges, align='center', alpha=0.5)
plt.xticks(np.arange(0, 24, 2))
plt.ylabel('Distribution')
plt.title('Cars by time of day')
plt.show()

#######################READ TSMC DATASET (NYC AND TOKYO)######################    
path="C:/Users/mayank/Downloads/ANIMESH/datasets/dataset_tsmc2014/"
trlist=[]
userlist=[]
for file in os.scandir(path):
    with open(file, 'r') as c_file:
        cread=csv.reader(c_file,delimiter='\t')
        lines=0
        for row in cread:
            userlist.append(float(row[0]))
            stx=row[7]
            st=stx.replace('+0000 ','')
            dt=datetime.strptime(st,'%a %b %d %H:%M:%S %Y')
            #coord=row[4].split('\t')
            #print(coord)
            trlist.append([float(row[4]),float(row[5]),dt])
            
from collections import Counter
lab=Counter(userlist).keys()
print(len(lab))

############################GO TRACKS BRAZIL DATASET##############################
path2="C:/Users/mayank/Downloads/ANIMESH/datasets/GPS Trajectory/"
import pandas as pd
df=pd.read_csv(path2+'go_track_trackspoints.csv')
trl=df.groupby('track_id').apply(lambda x: x[['latitude','longitude']].values.astype('float32').tolist())

trlist=[]
for i in trl:
    g=[]
    for j in range(len(i)):
        g.append([float(i[j][0]),float(i[j][1])])
    trlist.append(g)
    
######################ELIMINATE SPIRALS OR U-TURNS##########################
def roughrouting(trlist):
    for i in trlist:
        qs=i[0]
        qd=i[-1]
        for j in range(1,len(i)-1):
            if (distfind(qs,i[j])>distfind(qs,i[j+1])) or (distfind(qd,i[j])<distfind(qd,i[j+1])):
                del(i[j])
                
###############################MAP MATCHING####################################
##############################BING MAPS API####################################
modtrlist=[]
import requests
for i in trlist:
    d={}
    pointslist=[] 
    for j in i:
        d["latitude"]= float(j[0])
        d["longitude"]= float(j[1])
        pointslist.append(d.copy())
        
    response=requests.post('http://dev.virtualearth.net/REST/v1/Routes/SnapToRoad?interpolate=true&key=Arc88WQ28IRnvtfPwzKcNaZb_z40jjwR6ETFLYfIeXPRqmnl4DH8CX9RSnXvrDlm',json={"points":pointslist, "interpolate": "true"})
    if response.status_code==200:
        k=response.json()['resourceSets'][0]['resources'][0]['snappedPoints']
        #if len(k)<=3:
         #   print(k)
        trajectory=[]
        for i in k:
            trajectory.append([float(i['coordinate']['latitude']),float(i['coordinate']['longitude'])])
        if len(trajectory)>0:
            modtrlist.append(trajectory)
        #else:
         #   modtrlist.append(i)
 
#removing duplicates from matched trajectories
#not necessary if timestamped
b = list()
for sublist in modtrlist:
    if (sublist not in b) and len(sublist)>2:
        b.append(sublist)
        
##############################MAPBOX API####################################
from mapbox import MapMatcher
def mapmatch(coords):
    b=coords.tolist()
    if len(b)>100:
        matchlist=[]
        lim=0
        while 100*(lim+1)<len(b):
            matchlist.append(trajprocess(b[100*lim:100*(lim+1)]))
            lim+=1
        matchlist.append(trajprocess(b[100*lim:]))
        return matchlist
    else:
        return trajprocess(b)
            
def trajprocess(b):
    #b=[[39.9155,116.50661],[39.9156,116.49625]]
    '''
    b=[[13.418946862220764, 52.50055852688439],
       [13.419011235237122, 52.50113000479732],
       [13.419756889343262, 52.50171780290061],
       [13.419885635375975, 52.50237416816131],
       [13.420631289482117, 52.50294888790448]]
    '''
    line={
    "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": b}
    }
    service = MapMatcher(access_token="sk.eyJ1Ijoib3ZlcndhdGNoLTEyIiwiYSI6ImNrMjI5ZnFjdzBqZm4zcHBnN3N4NXgyYXgifQ.KZkzHpsZJPpax5dnCJPf6Q")
    response = service.match(line, profile='mapbox.driving')
    corrected = response.geojson()
    if corrected['code']=='Ok':
        print (corrected['features'][0]['properties']['confidence'])
        return corrected['features'][0]['properties']['matchedPoints']
    else:
        print (corrected['code'])
        return b
  
import numpy as np
def floatify(coordinates):
    coords=np.zeros(shape=(len(coordinates),2), dtype='float')
    k=0
    for i in coordinates:
        coords[k]=[float(i[0]),float(i[1])]
        k=k+1
    return coords

modtrlist1=[]
for i in range(len(trlist)):
    modtrlist1.append(mapmatch(floatify(trlist[i])))

###############################WRITE BACK PROCESSED DATA##########################    
import csv
with open("filename.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(trlist)
#np.savetxt('trajectories.csv',np.array(trlist),delimiter=',')
    
with open('trajmod.csv', mode='w') as dtfile:
    wr=csv.writer(dtfile, delimiter=',')
    for i in range(len(modtrlist)):
        wr.writerow(modtrlist[i])
    
####################READ PROCESSED DATA (MATCHED TRAJECTORIES)###################
import csv
trajlist=[]
ct=0
dataf=open("C:/Users/Animesh/Documents/vta research project/matched-trajectories.csv", 'r')
rd=csv.reader(dataf, delimiter=',', quoting=csv.QUOTE_NONE)
for row in rd:
    if ct%2==0:
        xt=[]
        i=0
        while i<len(row) and row[i]!="":
        #for i in range(0,len(row),2):
            try:
                lon=float(row[i][3:-1])
                lat=float(row[i+1][2:-3])
            except ValueError as e:
                print(ct, i)
                break
            xt.append([lat,lon])
            i+=2
        trajlist.append(xt)
    ct+=1
    
##########################BAD LOGIC AS WITHOUT TIMESTAMP#########################
#######################READ CELL SEQUENCES WITHOUT REPETITIONS###################
ctr=0
s=0
mt=[]
for i in range(len(x)):
    modtraj=[]
    s=s+len(x[i])
    while ctr<s:
        modtraj.append(y[ctr])
        #remove no change cell sequences
        while ctr+1<s and y[ctr]==y[ctr+1]:
            ctr+=1
        ctr+=1
    if len(modtraj)>1:
        mt.append(modtraj)
        
#########################REMOVE REPETITIONS IN SEQUENCES########################
b = list()
for sublist in modtrlist:
    if (sublist not in b) and len(sublist)>2:
        b.append(sublist)