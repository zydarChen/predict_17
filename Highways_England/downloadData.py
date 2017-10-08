# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup

soup = BeautifulSoup(open('he_2013.html'), 'lxml')
print soup.td