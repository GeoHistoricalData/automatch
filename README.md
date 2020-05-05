# automatch
A python tool for automatic image georeferencing.

## options
-i input image
-r reference (georeferenced) image
-o output image
--gcps if present, outputs the gcps as a shapefile next to your output image (optional)
--feature choose the feature detector. At the moment, the following are allowed: <sift|surf|orb|akaze|brisk>
--flann if present, use the FlannBasedMatcher. Otherwise, use the BFMatcher (Brute Force)
-tileX x dimension for the tiles (optional)
-tileY Y dimension for the tiles (optional)
-offsetX x dimension for the offset between tiles (optional)
-offsetY Y dimension for the offset between tiles (optional)

### example
```shell

python3 automatch.py -i your_awesome_image.jpg -r an_already_georeferenced_map.tif -o your_new_awesome_georeferenced_image.tif --gcps -tileX 512 -tileY 512 -offsetX 512 -offsetY 512

```

## Precomputation of keypoints and descriptors
Alternatively, you can save the keypoints and desctiptors as a custom file (sort of project file).
All the arguments used for the point detection will be saved.

### Usage
```shell
Usage: save_keypoints.py [OPTIONS] INPUTFILE OUTPUTFILE

  Compute the keypoints and desctiptors of an image and save them in a store
  file. All the arguments used for the point detection will be saved.

Options:
  -f, --feature TEXT             <sift|surf|orb|akaze|brisk>[-flann]
  -n, --n_features INTEGER       Number of features
  --tile <INTEGER INTEGER>...    Tile dimensions (width, height).
  --offset <INTEGER INTEGER>...  Offset as a pair (offsetX, offsetY)
  --help                         Show this message and exit.

```
### Example
```shell
python save_keypoints.py  some_image.jpg  store.pickle --tile 512 512 --offset 512 512 --feature orb
```

## Visualisation of keypoints and descriptors
Now that they have been saved, you can visualise your keypoints store in the custom file (sort of project file).

### example
```shell

python3 load_keypoints.py your_output_project_file

```

## Run automatch with the stored keypoints and descriptors
You can also run automatch with the stored keypoints. That reduces the number of required arguments:

### options
-ki input project file
-kr reference project file
-o output image
--gcps if present, outputs the gcps as a shapefile next to your output image (optional)
--flann if present, use the FlannBasedMatcher. Otherwise, use the BFMatcher (Brute Force)

### example
```shell

python3 automatch.py -ki your_awesome_project_file -kr an_already_georeferenced_map_project_file -o your_new_awesome_georeferenced_image.tif --gcps --flann


```

