###################################
#
#	Written by: Nitay Hason
#	Email: nitay.has@gmail.com
#
###################################
from flask import Flask,render_template

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import decimal
import operator
import datetime

from mydb import Dynamodb

app = Flask(__name__,template_folder='templates',static_url_path='/static')
dynamodb = Dynamodb()
people_table = dynamodb.getTable('People')
states_data_table = dynamodb.getTable('StatesData')

emotions_arr={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Surprise",6:"Natural"}

@app.route("/")
@app.route('/index')
def index():
    user = {'username': 'Nitay Hason'}
    items = states_data_table.scan()
    # print(items["Items"])
    ids = set( dic["PersonId"] for dic in items["Items"])
    sdata = []
    for id in ids:
        person = [item for item in items["Items"] if item["PersonId"]==id]
        # print(person[:-1])
        total_emo={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        last_update = 0
        for data in person:
            total_emo[int(data["State"])]+=data["Duration"]

        max_emotion = int(max(total_emo.iteritems(), key=operator.itemgetter(1))[0])

        last_update = person[-1]["StateDataId"]
        last_seen = person[-1]["State"]
        sdata.append({
            'picture': './static/students/%s.jpg' % id,
            'id': id,
            'total_state': emotions_arr[max_emotion],
            'last_update': datetime.datetime.fromtimestamp(float(last_update)).strftime("%d/%m/%Y %H:%M:%S"),
            'last_seen': emotions_arr[int(last_seen)]
        })

    # sdata = [
    #     {
    #         'picture': './static/students/305691347.jpg',
    #         'id': '305691347',
    #         'total_state': 'Sad',
    #         'last_update': '21/3/2018 18:34:23',
    #         'last_seen': 'Sad'
    #     },
    #     {
    #         'picture': './static/students/343928463.jpg',
    #         'id': '343928463',
    #         'total_state': 'Happy',
    #         'last_update': '21/3/2018 18:34:23',
    #         'last_seen': 'Angry'
    #     }
    # ]
    return render_template('./index.html', user=user, sdata=sdata)

if __name__ == "__main__":
    app.run()
