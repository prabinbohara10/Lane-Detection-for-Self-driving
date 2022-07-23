# Lane-Detection-for-Self-driving
Initial approach is simply by using the canny edge detection algorithm.

1) Converting the 3 channel image to gray scale.
2) Using the gaussian filter to reduse noise and smoothening image.
  Basically, it works by averaging the pixel around particular pixel and changing it's value.
3) Applying canny edge detection
  Firstly, computing the gradient as a series of white pixel.
  Then tracing the strong gradient as a series of white pixel.
4) Calculating the region of interest.
  This can be done by localizing the position of vechile we're training on. We can find out the detected nearest line from the point of localization.
  In my sample testing, I have used the hand coded ROI where white pixel denotes my region.
5) Applying bitwise_and to the ROI coded image and the original image so that other objects in the frame are discarded.
6) Hough transform to find hough space.
  For each pixel with white color, there exists a line in a hough space(Rho vs theta)
  ![image](https://user-images.githubusercontent.com/42064827/180612196-ea4ede61-b8dc-4808-a27a-ab625331ea28.png)
  ![image](https://user-images.githubusercontent.com/42064827/180612228-9e137a3d-a10a-46c1-904b-bced7e9d5a48.png)
  
  In cartesian co-ordinate,
  
  
  ![image](https://user-images.githubusercontent.com/42064827/180612114-cd388589-45bf-4111-9c80-d5138f91ab0c.png)
