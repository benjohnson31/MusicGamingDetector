# MusicGamingDetector
Project Description:
There are three data files come from a streaming music company.  The data files summarize a single users streaming music consumption / usage on the dates mentioned in the filename.  
There are two other data sets (1 test / 1 validation) that include marked examples of anomalous usage.  
They were marked by hand, so should not be taken as "fact" - but simple very probably examples.

Column descriptions
The userID is a database surrogate for a GUID.\n
IndieToMajorRatio is a ratio of usage on content from "independent labels" / "major labels".  The theory being that independent music
is more likely targeted for abuse\n
totUsage = the users total usage count on that day \n
MaxSingleAlbumUse = Of all albums played by the user on that day, the stream count of the album achieving the most usage by a user\n
MaxSingleTrackUse = Of all tracks played by the user on that day, the stream count of the track achieving the most usage by a user\n
MaxSingleArtistUse = Of all Artists played by the user on that day, the stream count of the Artist achieving the most usage by a user\n

This data was used to detect anomalous users by way of a gaussian distribution model.  I no longer have access to the source data to 
grab new metrics / aggregate differently, but would perhaps choose more / different measures if it were to be done again.

The .py file can be used as a module in python3.
required packages include numpy and scipy
The data sets are imported in the comments at the bottom of the .py file.
The method (trainData()) that is called that creates the boundary parameters (epsilon / mu)  is called in the comments.
You can test the output epsilon / mu using the test data method (testThreshold())

If you got this far, amazing, I'd be curious what you think of the model
