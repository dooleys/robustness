#!/bin/bash

# assumes the argument is a file where each line is a path to an .MP4 file
# e.g., the file mp4s.txt could  look like
#      foo/bar/baz/movie1.MP4
#      foo/bar/movie2.MP4
#      ...
# and this bash script witll take the frame that is 1 second into each video
# writing it as
#      foo/bar/baz/movie1.png
#      foo/bar/movie2.png
#      ...



mv4_files=$1

while read p; do
  stringarray=($p)
  echo ${stringarray[1]}
  ffmpeg -accurate_seek -ss 1 -i ${stringarray[1]} -frames:v 1 $(sed 's/.MP4/.png/' <<< ${stringarray[1]})
done < $mv4_files
