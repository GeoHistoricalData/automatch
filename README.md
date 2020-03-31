# automatch
A python tool for automatic image georeferencing

## options
-i input image
-r reference (georeferenced) image
-o output image
--gcps if present, outputs the gcps as a shapefile next to your output image (optional)
-tileX x dimension for the tiles (optional)
-tileY Y dimension for the tiles (optional)
-offsetX x dimension for the offset between tiles (optional)
-offsetY Y dimension for the offset between tiles (optional)

## example
```shell

python3 automatch.py -i your_awesome_image.jpg -r an_already_georeferenced_map.tif -o your_new_awesome_georeferenced_image.tif --gcps -tileX 512 -tileY 512 -offsetX 512 -offsetY 512

```