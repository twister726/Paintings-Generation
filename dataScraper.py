import urllib.request
import requests
def parseLine(line):
    pass
if __name__=='__main__':
  with open('dataset.csv', 'r') as f:
      line = f.readline()
      count = 0
      total = 0
      for i, line in enumerate(f):
          total +=1
          vals = line.split(',')
          url = vals[1]
          #url = url[:-4] + "001" + url[-4:]

          try:
            r = requests.get(url)
            img_url = r.text.split("<img src=")[1].split(" ")[0][1:-1]
            path = "./data/" + img_url.split("/")[-1]

            urllib.request.urlretrieve(img_url,path) 

#            if imgr.status_code == 200:
#                count += 1
#                print("{} valid of {}".format(count,total))
#                print("url: {}\nstatus: {}".format(img_url, r.status_code))
#                path = "./data/" + img_url.split("/")[-1]
#                with open(
          except :
            pass
