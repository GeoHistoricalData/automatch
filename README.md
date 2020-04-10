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

### example
```shell

python3 save_keypoints.py -i your_awesome_image.jpg -o your_output_project_file -tileX 512 -tileY 512 -offsetX 512 -offsetY 512 --feature orb

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

