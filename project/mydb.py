###################################
#
#	Written by: Nitay Hason
#	Email: nitay.has@gmail.com
#
###################################
import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import json
import decimal

# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
	def default(self, o):
		if isinstance(o, decimal.Decimal):
			if abs(o) % 1 > 0:
				return float(o)
			else:
				return int(o)
		return super(DecimalEncoder, self).default(o)

class Dynamodb:
	def __init__(self):
		self.dynamodb = boto3.resource('dynamodb')

	def getTable(self, tablename):
		return self.dynamodb.Table(tablename)

	def getItem(self, table, key):
		resp = None
		try:
			response = table.get_item(
				Key = key
				)
		except ClientError as e:
			print(e.response['Error']['Message'])
		else:
			resp = json.dumps(response["Item"], indent=4, cls=DecimalEncoder)
			resp = json.loads(resp)

		return resp

	def putItem(self,cur_table,item):
		cur_table.put_item(
		   Item=item
		)
