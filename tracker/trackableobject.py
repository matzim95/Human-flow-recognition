class TrackableObject:
	def __init__(self, objectID, centroid): #class of one unique tracking object, storing information about object ID,
											# its centroid and if it has been counted
		self.objectID = objectID
		self.centroids = [centroid]
		self.counted = False