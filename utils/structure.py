class Node(object):
	def __init__(self, val, type, layer=1):
		self.val = val
		# not leaf node.
		self.type = type
		self.children = []
		self.layer = layer

