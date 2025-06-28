import json
from glob import glob
files = glob('*.json')
results = {}
for fil in files:
    try:
        data = json.load(open(fil, 'rb'))
        crop = 0
        bbox = 0
        crop_sam = 0
        total =0 
        for key, values in data.items():
            for vals in values:
                if len(vals)==4:
                    gt = vals[0]
                    if gt == vals[1]:
                        crop+=1
                    if gt == vals[2]:
                        bbox+=1
                    if gt == vals[3]:
                        crop_sam+=1
                    total+=1
        results[fil] = {'crop':crop/total, 'bbox':bbox/total, 'crop_sam': crop_sam/total}
    except:
        pass

for key, value in results.items():
    print(key, value)
    print()
