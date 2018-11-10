# -*- coding: utf-8 -*-

import xml.dom.minidom
from xml.dom.minidom import parse
import os

DOMTree = xml.dom.minidom.parse(r'sawsdl-tc3.xml')
collection = DOMTree.documentElement

Requests = collection.getElementsByTagName('request')

for request in Requests:
    RQUri = request.getElementsByTagName('uri')[0].firstChild.data #the childs of an Element contains the text between the start and the end label
                                                              #like '\n' + element + '\n' + element,so that it may not be the first child of childNodes
    RQ = RQUri.split("/")[-1].split('#')[0]
    ratings = request.getElementsByTagName('ratings')[0]

    offers = ratings.getElementsByTagName('offer')
    dict = {}
    for offer in offers:
        serviceUri = offer.getElementsByTagName('uri')[0].firstChild.data
        service = serviceUri.split('/')[-1].split('#')[0]
        value = offer.getElementsByTagName('value')
        if len(value) == 0:
            value = offer.getElementsByTagName('relevant')
        value = value[0].firstChild.data
        dict[service] = value
    sortedRes = sorted(dict.items(),key = lambda k : k[1],reverse = True)

    with open(os.path.join('./reqRelevance/',RQ),'w') as f:
        for key,value in sortedRes:
            f.writelines(key + "\t" + "{0}\n".format(value))
