# semanticencoder
A simple script to create semantic 2D encodings of texts for later display in 2D scatterplot

### Dependencies
See requirements.txt

Note, if you don't want to produce your own examples, just use exampledata.json as example output of this process
### Running instructions
1. Download the Universal Sentence Encoder https://tfhub.dev/google/universal-sentence-encoder-lite/2 (this is the smallest model and still gives good results, we can use a larger model if required)
2. Update line 25 of embeddings_to_plot.py to point to the directory of the encoder (contents of this folder should be /assets, /variables etc)
3. Install all python dependencies in requirements.txt
4. Run embeddings_to_plot.py - updating the script "texts" array with your own examples

The script will display two graphs (slightly different params of same data to allow comparison) and write an exampledata.json file (you may wish to specify your own output location at line 155) 

Use the exampledata.json file to build a pretty scatter plot. 

In it's final form, we will only supply the "doc_id" for each x,y node. The full document will be stored in elasticsearch and then can be retrieved via doc_id for display upon selection of a node. I may also look at producing aggregations of data in a side drawer if multiple nodes are selected - such as the significant common terms (elasticsearch can provide this if you give it an array of docids). 
