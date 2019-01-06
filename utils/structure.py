class Node(object):
	def __init__(self, val, type):
		self.val = ""
		# not leaf node.
		self.type = type
		self.children = {}

