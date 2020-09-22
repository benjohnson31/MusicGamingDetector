# MusicGamingDetector

[**Problem Statement & Project Summary**](https://docs.google.com/presentation/d/15OxRR7H1gkSy3UKbzwC-6b90n9tONWeCWuAs_4Wf42I/edit#slide=id.g99f8faf5dc_0_93)

**Overview of Repository:**\
There are three data files from a streaming music company that summarize a user streaming music consumption / usage on the dates mentioned in the filename.  
The Dec10 data is a 10,000 user sample (out of total 1 million or so active users) for a single day used to train the model.  
There are two other data sets (1 test / 1 validation) that include marked examples of anomalous usage. 
They were marked by hand, so should not be taken as "fact" - but instead as very likely examples marked in a short amount of time (I'd like to think decent quality).
There is a python script that trains and tests the model

**Data Descriptions:**\
Each row is a single users activity for a day.

Column descriptions:\
The userID is a database surrogate for a GUID.\
IndieToMajorRatio is a ratio of usage on content from "independent labels" / "major labels".  The theory being that music from independent labels
is more likely targeted for abuse\
totUsage = the users total usage count on that day \
MaxSingleAlbumUse = Of all albums played by the user on that day, the stream count of the album achieving the most usage by a user\
MaxSingleTrackUse = Of all tracks played by the user on that day, the stream count of the track achieving the most usage by a user\
MaxSingleArtistUse = Of all Artists played by the user on that day, the stream count of the Artist achieving the most usage by a user

**Model Description**\
If you didn't read the linked presentation, this data was used to detect anomalous users ('abusive' usage) by way of a gaussian distribution model.
I no longer have access to the source data to grab new metrics / aggregate differently, but would perhaps choose more / different measures if it were to be done again.

Note - the training data set includes all of the known anomalies - there is a feature if you want to trim the dataset by # standard deviations to get what is considered 'normal'
The .py file can be used as a module in python3.
required packages include numpy and scipy
The data sets are imported in the comments at the bottom of the .py file.
The method (trainData()) that is called that creates the boundary parameters (epsilon / mu)  is called in the comments.
You can test the output epsilon / mu using the test data method (testThreshold())

If you got this far - into the linked presentation, or the model / data itself - thank you, you are amazing.  
I'd be curious what you think of the model
