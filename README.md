# ImageMorphingApp
The image morphing app is able to morph a pair of images with customized weight of the image pair. 
The following shows what the program produce as a result.  
<img src="/Tiger2Color.jpg" width="140"/>
<img src="/color_results/frame040.jpg" width="140">
<img src="/color_results/frame050.jpg" width="140">
<img src="/color_results/frame055.jpg" width="140">
<img src="/color_results/frame062.jpg" width="140">
<img src="/color_results/frame070.jpg" width="140">
<img src="/WolfColor.jpg" width="140">
With the video option turned on, it can also compile the image set into a mp4 video.
The image pair are seperated into matching triangle segments and affine transform and interpolation is applied on the pixels in each triangle. The rendering speed of a single destination image is about 2 seconds.
