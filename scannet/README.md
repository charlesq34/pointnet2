### ScanNet Data

Original dataset website: <a href="http://www.scan-net.org/">http://www.scan-net.org/</a>

You can get our preprocessed data at <a href="https://shapenet.cs.stanford.edu/media/scannet_data_pointnet2.zip">here (1.72GB)</a> and refer to the code in `scannet_util.py` for data loading. Note that the virtual scan data is generated on the fly from our preprocessed data.

Some code we used for scannet preprocessing is also included in `preprocessing` folder. You have to download the original ScanNet data and make small modifications in paths in order to run them.

Note: To use ScanNetV2 data, change the tsv file to `scannetv2-labels.combined.tsv` and also update `scannet_util.py` to read the raw class and NYU40 names in the right columns (shifted by 1 compared to the V1 tsv).
