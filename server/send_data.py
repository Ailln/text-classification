import http.client
import urllib.parse

with open("./data/THUCNews-5_2000/test/体育-76.txt", "r", encoding="utf-8") as f_file:
    input_data = ""
    for line in f_file:
        line = line.strip().replace(" ", "")
        input_data += line

params = urllib.parse.urlencode({"text": input_data})

conn = http.client.HTTPConnection("127.0.0.1:5000")
headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
conn.request("POST", "/text_classification", body=params, headers=headers)
response = conn.getresponse()
print(response.read().decode("utf-8"))
conn.close()
