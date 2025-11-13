# Tracks

Note for Train Track 1, Test Track 2, and Test Track 3. These are community tracks not official tracks. 
Therefore there is no centerline csv file to be used by the simulator. Instead, a script must be used to convert the YAML metadata
and the image and generate a CSV file. This can be done by loading the yaml into python, extracting track boundaries,
computing the centerline, smoothing the curve, converting pixels to coordinates, and exporting as csv.

This script can either be created in python, or you can try and find a pre-made on online.

In general, you can use any map in the simulator launce files as follows
```
 <arg name="map" default="$(find f1tenth_maps)/maps/f1_aut.yaml"/>
  <node pkg="map_server" name="map_server" type="map_server" args="$(arg map)"/>
```
