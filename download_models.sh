#!/bin/bash

url='https://style-transfer.s3-us-west-2.amazonaws.com/vgg19.caffemodel'

echo 'Downloading a truncated version of the VGG-19 pre-trained model.'
echo 'See http://www.robots.ox.ac.uk/~vgg/research/very_deep/.'

cd models
curl -L "$url" > vgg19.caffemodel.download
mv vgg19.caffemodel.download vgg19.caffemodel
