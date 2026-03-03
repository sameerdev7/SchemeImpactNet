import requests
from bs4 import BeautifulSoup

url = "https://nreganarep.nic.in/netnrega/misreport4.aspx"
headers = {"User-Agent": "Mozilla/5.0"}

session = requests.Session()
res = session.get(url, headers=headers)

soup = BeautifulSoup(res.text, "html.parser")

viewstate = soup.find("input", {"name": "__VIEWSTATE"})["value"]
eventvalidation = soup.find("input", {"name": "__EVENTVALIDATION"})["value"]

print("Extracted tokens ✔")
