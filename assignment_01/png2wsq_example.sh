#!/bin/bash

# Convert $1.png to a grayscale image and save as $1.gray
magick "$1.png" -depth 8 -type Grayscale -compress none -colorspace Gray -strip "$1.gray"

# Create WSQ from the RAW image, -raw_in W,H,bits,DPI, just keep in mind that pcasys needs more than 300
cwsq 0.75 wsq "$1.gray" -raw_in 600,600,8,500

# Compute minutiae on WSQ
mindtct "$1.wsq" "$1"
