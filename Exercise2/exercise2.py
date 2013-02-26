# coding: utf-8

## Required imports.
import cv2, numpy as np, os

# Exercise 2.1 ----------------------------------------------------------------------

# TODO finish this for exercise 2.1.a
def ComputeGradients(image):

  # Define the kernels
  # TODO
  dxKernel = np.array([[1]], dtype='float32')
  dyKernel = np.array([[1]], dtype='float32')
  dx2Kernel = np.array([[1]], dtype='float32')
  dy2Kernel = np.array([[1]], dtype='float32')
  laplacianKernel = np.array([[1]], dtype='float32')

  # Apply the kernels to the image
  dx = cv2.filter2D(image, cv2.CV_32F, dxKernel)
  dy = cv2.filter2D(image, cv2.CV_32F, dyKernel)
  dx2 = cv2.filter2D(image, cv2.CV_32F, dx2Kernel)
  dy2 = cv2.filter2D(image, cv2.CV_32F, dy2Kernel)
  laplacian = cv2.filter2D(image, cv2.CV_32F, laplacianKernel)

  # Calculate the gradient magniture
  # TODO
  gradMag = np.zeros_like(image)
  
  return dx, dy, gradMag, dx2, dy2, laplacian

# TODO finish this for exercise 2.1.b
def ComputeEdges(dx, dy, gradientMagnitude):

  # TODO: Threshold the images at magnitude over 0.075.
  dxEdges = dx
  dyEdges = dy
  gradientMagnitudeEdges = gradientMagnitude
  
  return dxEdges, dyEdges, gradientMagnitudeEdges

# TODO finish this for exercise 2.1.c
def ComputeCanny(image):

  # TODO: Compute Canny edges (using second-order gradient).

  return image

# Exercise 2.2 ----------------------------------------------------------------------

# TODO finish for exercise 2.2
def scaleSpaceEdges(image):

  # calculating gaussians of an image
  gauss1 = image;
  gauss2 = image;
  gauss3 = image;

  # calculate the edges using code from previous exercises
  dx1, dy1, gradMag1 = ComputeGradients(image)[:3]
  dx2, dy2, gradMag2 = ComputeGradients(gauss1)[:3]
  dx3, dy3, gradMag3 = ComputeGradients(gauss2)[:3]
  dx4, dy4, gradMag4 = ComputeGradients(gauss3)[:3]
  
  dxEdges1, dyEdges1, gradEdges1 = ComputeEdges(dx1, dy1, gradMag1)
  dxEdges2, dyEdges2, gradEdges2 = ComputeEdges(dx2, dy2, gradMag2)
  dxEdges3, dyEdges3, gradEdges3 = ComputeEdges(dx3, dy3, gradMag3)
  dxEdges4, dyEdges4, gradEdges4 = ComputeEdges(dx4, dy4, gradMag4)
  
  # Return ALL the things
  return gauss1, gauss2, gauss3, dxEdges1, dyEdges1, gradEdges1, dxEdges2, dyEdges2, gradEdges2, dxEdges3, dyEdges3, gradEdges3, dxEdges4, dyEdges4, gradEdges4

# Exercise 2.3 ----------------------------------------------------------------------

# TODO finish for exercise 2.3
def DifferenceOfGaussiansLaplacian(gauss1, gauss2, gauss3):

  # Calculate the Laplacian of gauss1 and gauss2
  # TODO
  Laplacian1 = np.zeros_like(gauss1)
  Laplacian2 = np.zeros_like(gauss1)

  # Calculate the difference of gaussians
  # TODO
  DoG1 = np.zeros_like(gauss1)
  DoG2 = np.zeros_like(gauss1)
  
  return Laplacian1, Laplacian2, DoG1, DoG2


## ---------------------------------------------------------------------
## ---------------------------------------------------------------------
## YOU DON'T NEED TO LOOK PAST HERE
## (Please don't, the code is pretty ugly)
## ---------------------------------------------------------------------
## ---------------------------------------------------------------------



# We define some utility functions for normalising and writing out images

# Writing out the gradient images
def WriteOutGradients(dx, dy, gradMag, dx2, dy2, laplacian):

  # Normalising the images so they can be saved
  dxOut = normalise(dx);
  cv2.imwrite(os.path.expanduser("dx.png"), np.uint8(255 * dxOut))
  dyOut = normalise(dy);
  cv2.imwrite(os.path.expanduser("dy.png"), np.uint8(255 * dyOut))
  gradMagOut = normalise_pos(gradMag);
  cv2.imwrite(os.path.expanduser("gradMag.png"), np.uint8(255 * gradMagOut))
  dx2Out = normalise(dx2);
  cv2.imwrite(os.path.expanduser("dx2.png"), np.uint8(255 * dx2Out))
  dy2Out = normalise(dy2);
  cv2.imwrite(os.path.expanduser("dy2.png"), np.uint8(255 * dy2Out))
  laplacianOut = normalise(laplacian);
  cv2.imwrite(os.path.expanduser("laplacian.png"), np.uint8(255 * laplacianOut))

  cv2.putText(dxOut, 'dx', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(dyOut, 'dy', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(dx2Out, 'dx2', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(dy2Out, 'dy2', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(gradMagOut, 'Gradient Magnitude', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(laplacianOut, 'Laplacian', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  
  row1 = np.concatenate((dxOut, dyOut, gradMagOut), axis = 1)
  row2 = np.concatenate((dx2Out, dy2Out, laplacianOut), axis = 1)

  combinedEdges = np.concatenate((row1, row2), axis = 0)

  ## Save images to disk for comparison.
  cv2.imwrite(os.path.expanduser("combinedGradients.png"), np.uint8(255 * combinedEdges))
  
  return combinedEdges

# Writing out the edge images
def OutputEdges(dxEdges, dyEdges, gradEdges, cannyEdges):
  cv2.imwrite(os.path.expanduser("dxedges.png"), np.uint8(255 * dxEdges))
  cv2.imwrite(os.path.expanduser("dyedges.png"), np.uint8(255 * dyEdges))
  cv2.imwrite(os.path.expanduser("gradmagedges.png"), np.uint8(255 * gradEdges))

  cv2.putText(dxEdges, 'dx edges', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(dyEdges, 'dy edges', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  cv2.putText(gradEdges, 'grad mag edges', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))
  # convert to a floating point first (as canny returns uint8)
  cannyEdges = cannyEdges / 255.0
  cv2.putText(cannyEdges, 'Canny', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (1.0))

  combinedEdges = np.concatenate((dxEdges, dyEdges, gradEdges, cannyEdges), axis=1)

  # Save the edge images
  cv2.imwrite(os.path.expanduser("combinedEdges.png"), np.uint8(255 * combinedEdges))
  
  return combinedEdges
  
def OutputScaleSpaceEdges(dxEdges1, dyEdges1, gradEdges1, dxEdges2, dyEdges2, gradEdges2, dxEdges3, dyEdges3, gradEdges3, dxEdges4, dyEdges4, gradEdges4):

  # Normalising and writing out to disk
  cv2.putText(dxEdges1, 'sigma = 0', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(dxEdges2, 'sigma = 1', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(dxEdges3, 'sigma = 1.6', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(dxEdges4, 'sigma = 2.56', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)

  cv2.putText(dxEdges1, 'dx edges', (170,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(dyEdges1, 'dy edges', (170,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(gradEdges1, 'Grad Mag Edges', (100,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)

  # Concatenating the output images
  separator = np.ones((dxEdges.shape[0], 2))
  
  row1 = np.concatenate((dxEdges1, separator, dyEdges1, separator, gradEdges1), axis = 1)
  row2 = np.concatenate((dxEdges2, separator, dyEdges2, separator, gradEdges2), axis = 1)
  row3 = np.concatenate((dxEdges3, separator, dyEdges3, separator, gradEdges3), axis = 1)
  row4 = np.concatenate((dxEdges4, separator, dyEdges4, separator, gradEdges4), axis = 1)

  scaleSpace = np.concatenate((row1, row2, row3, row4), axis = 0)
  cv2.imwrite(os.path.expanduser("scaleSpace.png"), np.uint8(255 * scaleSpace))
  
  return scaleSpace
  
def OutputDoGLap(Laplacian1, Laplacian2, DoG1, DoG2):

  # Normalising and writing out to disk
  Laplacian1 = normalise(Laplacian1)
  Laplacian2 = normalise(Laplacian2)

  DoG1 = normalise(DoG1)
  DoG2 = normalise(DoG2)
  
  cv2.imwrite(os.path.expanduser("Laplacian1.png"), np.uint8(255 * Laplacian1))
  cv2.imwrite(os.path.expanduser("Laplacian2.png"), np.uint8(255 * Laplacian2))
  cv2.imwrite(os.path.expanduser("DoG1.png"), np.uint8(255 * DoG1))
  cv2.imwrite(os.path.expanduser("DoG2.png"), np.uint8(255 * DoG2))

  cv2.putText(Laplacian1, 'Laplacian', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)
  cv2.putText(DoG1, 'Difference of Gaussians', (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1.0)

  row1 = np.concatenate((Laplacian1, DoG1), axis = 1)
  row2 = np.concatenate((Laplacian2, DoG2), axis = 1)

  laplacianVsDoG = np.concatenate((row1, row2), axis = 0)
  cv2.imwrite(os.path.expanduser("laplacianVsDoG.png"), np.uint8(255 * laplacianVsDoG))
  
  return laplacianVsDoG

# Normalisation
def normalise_pos(image):
  'Normalises the image so that the values lie from 0 to 1.'

  if image.max() != 0:
    output = image / image.max()
  else:
    output = np.zeros_like(image)
  output = output
  
  return output
def normalise(image):
  'Normalises the image so that the values lie from 0 to 1.'

  if image.min() != 0 or image.max() != 0:
    output = image * 0.5 / max(image.max(), -image.min())
  else:
    output = np.zeros_like(image)
  output = output + 0.5
  
  return output


## ---------------------------------------------------------------------

def show(name, im):
  if im.dtype == np.complex128:
    raise Exception("OpenCV can't operate on complex valued images")
  cv2.namedWindow(name)
  cv2.imshow(name, im)
  cv2.waitKey(1)

if __name__ == '__main__':

  verbose = True
  
  ## Start background thread for event handling of windows.
  #if verbose:
  #  cv2.namedWindow("image")
  #  cv2.namedWindow("result")
  #  cv2.startWindowThread()
  
  ## Read in example image (greyscale, float, resize).
  image = cv2.imread("input/lena.png", 0)  
  #image = cv2.imread("C:/openCV 2.3.1/opencv/samples/c/lena.jpg",0)  
  imageUint8 = cv2.pyrDown(image);

  # See the image we're working on
  cv2.imwrite(os.path.expanduser("image-input.png"), imageUint8)

  imageF32 = np.array(imageUint8 / 255.0, dtype='float32');

  ## Apply them to an image
  dx, dy, gradMag, dx2, dy2, laplacian = ComputeGradients(imageF32)

  # Write them to disc
  gradients = WriteOutGradients(dx, dy, gradMag, dx2, dy2, laplacian)
  
  show("Gradients", gradients)
 
  # Second part of the exercise, actually getting the edges
  dxEdges, dyEdges, gradMagEdges = ComputeEdges(dx, dy, gradMag)  

  # A more exciting edge detector
  canny = ComputeCanny(imageUint8)

  combinedEdges = OutputEdges(dxEdges, dyEdges, gradMagEdges, canny)
  
  show("Edges", combinedEdges)

  # Exercise 2.2 - edge detection in scale space
  gauss1, gauss2, gauss3, dxEdges1, dyEdges1, gradEdges1, dxEdges2, dyEdges2, gradEdges2, dxEdges3, dyEdges3, gradEdges3, dxEdges4, dyEdges4, gradEdges4 = scaleSpaceEdges(imageF32)
  
  scaleSpaceEdges = OutputScaleSpaceEdges(dxEdges1, dyEdges1, gradEdges1, dxEdges2, dyEdges2, gradEdges2, dxEdges3, dyEdges3, gradEdges3, dxEdges4, dyEdges4, gradEdges4)
  
  show("Scale space edges", scaleSpaceEdges)

  # Exercise 2.3 - Approximating Laplacian as difference of Gaussians
  Laplacian1, Laplacian2, DoG1, DoG2 = DifferenceOfGaussiansLaplacian(gauss1, gauss2, gauss3)
  
  doglap = OutputDoGLap(Laplacian1, Laplacian2, DoG1, DoG2)
  
  show("DoG vs Laplacian", doglap)
  
  # Check answers
  # 2.1.a dx
  dx_student = cv2.imread(os.path.expanduser("dx.png"), 0) / 255.0
  dx_correct = cv2.imread(os.path.expanduser("answer-images/dx.png"), 0) / 255.0
  if dx_student.shape == dx_correct.shape and np.all(np.abs(dx_student - dx_correct) < 0.01):
    print "Exercise 2.1.a: dx CORRECT"
  else:
    print "Exercise 2.1.a: dx INCORRECT"
    dx_inv = cv2.imread(os.path.expanduser("answer-images/dx-inv.png"), 0) / 255.0
    if dx_student.shape != dx_correct.shape:
      print "    The size is incorrect. The result should be the same size as the input"
    elif np.all(np.abs(dx_student - dx_inv) < 0.01):
      print "    Check if the direction of the gradient kernel is correct"
  # 2.1.a dy
  dy_student = cv2.imread(os.path.expanduser("dy.png"), 0) / 255.0
  dy_correct = cv2.imread(os.path.expanduser("answer-images/dy.png"), 0) / 255.0
  if dy_student.shape == dy_correct.shape and np.all(np.abs(dy_student - dy_correct) < 0.01):
    print "Exercise 2.1.a: dy CORRECT"
  else:
    print "Exercise 2.1.a: dy INCORRECT"
    dy_inv = cv2.imread(os.path.expanduser("answer-images/dy-inv.png"), 0) / 255.0
    if dy_student.shape != dy_correct.shape:
      print "    The size is incorrect. The result should be the same size as the input"
    elif np.all(np.abs(dy_student - dy_inv) < 0.01):
      print "    Check if the direction of the gradient kernel is correct"
  # 2.1.a dx^2
  dx2_student = cv2.imread(os.path.expanduser("dx2.png"), 0) / 255.0
  dx2_correct = cv2.imread(os.path.expanduser("answer-images/dx2.png"), 0) / 255.0
  if dx2_student.shape == dx2_correct.shape and np.all(np.abs(dx2_student - dx2_correct) < 0.01):
    print "Exercise 2.1.a: dx^2 CORRECT"
  else:
    print "Exercise 2.1.a: dx^2 INCORRECT"
    dx2_inv = cv2.imread(os.path.expanduser("answer-images/dx2-inv.png"), 0) / 255.0
    dx2_blurred = cv2.imread(os.path.expanduser("answer-images/dx2-blurred.png"), 0) / 255.0
    if dx2_student.shape != dx2_correct.shape:
      print "    The size is incorrect. The result should be the same size as the input"
    elif np.all(np.abs(dx2_student - dx2_inv) < 0.01):
      print "    Check if the polarity of the kernel is correct"
    elif np.all(np.abs(dx2_student - dx2_blurred) < 0.01):
      print "    You are using 'a' directional edge detector, but not the second finite difference operator. Read the notes carefully."
  # 2.1.a dy^2
  dy2_student = cv2.imread(os.path.expanduser("dy2.png"), 0) / 255.0
  dy2_correct = cv2.imread(os.path.expanduser("answer-images/dy2.png"), 0) / 255.0
  if dy2_student.shape == dy2_correct.shape and np.all(np.abs(dy2_student - dy2_correct) < 0.01):
    print "Exercise 2.1.a: dy^2 CORRECT"
  else:
    print "Exercise 2.1.a: dy^2 INCORRECT"
    dy2_inv = cv2.imread(os.path.expanduser("answer-images/dy2-inv.png"), 0) / 255.0
    dy2_blurred = cv2.imread(os.path.expanduser("answer-images/dy2-blurred.png"), 0) / 255.0
    if dy2_student.shape != dy2_correct.shape:
      print "    The size is incorrect. The result should be the same size as the input"
    elif np.all(np.abs(dy2_student - dy2_inv) < 0.01):
      print "    Check if the polarity of the kernel is correct"
    elif np.all(np.abs(dy2_student - dy2_blurred) < 0.01):
      print "    You are using 'a' directional edge detector, but not the second finite difference operator. Read the notes carefully."
      
  ## I warned you about the ugly code
  
  dxedges_student = cv2.imread(os.path.expanduser("dxedges.png"), 0) / 255.0
  dxedges_correct = cv2.imread(os.path.expanduser("answer-images/dxedges.png"), 0) / 255.0
  if dxedges_student.shape == dxedges_correct.shape and np.all(np.abs(dxedges_student - dxedges_correct) < 0.01):
    print "Exercise 2.1.b: dx edges CORRECT"
  else:
    print "Exercise 2.1.b: dx edges INCORRECT"
  dyedges_student = cv2.imread(os.path.expanduser("dyedges.png"), 0) / 255.0
  dyedges_correct = cv2.imread(os.path.expanduser("answer-images/dyedges.png"), 0) / 255.0
  if dyedges_student.shape == dyedges_correct.shape and np.all(np.abs(dyedges_student - dyedges_correct) < 0.01):
    print "Exercise 2.1.b: dy edges CORRECT"
  else:
    print "Exercise 2.1.b: dy edges INCORRECT"
  gradmagedges_student = cv2.imread(os.path.expanduser("gradmagedges.png"), 0) / 255.0
  gradmagedges_correct = cv2.imread(os.path.expanduser("answer-images/gradmagedges.png"), 0) / 255.0
  if gradmagedges_student.shape == gradmagedges_correct.shape and np.all(np.abs(gradmagedges_student - gradmagedges_correct) < 0.01):
    print "Exercise 2.1.b: gradient magnitude edges CORRECT"
  else:
    print "Exercise 2.1.b: gradient magnitude edges INCORRECT"
    
  # No check for Canny
  
  scaleSpace_student = cv2.imread(os.path.expanduser("scaleSpace.png"), 0) / 255.0
  scaleSpace_correct = cv2.imread(os.path.expanduser("answer-images/scaleSpace.png"), 0) / 255.0
  if scaleSpace_student.shape == scaleSpace_correct.shape and np.all(np.abs(scaleSpace_student - scaleSpace_correct) < 0.01):
    print "Exercise 2.2: CORRECT"
  else:
    print "Exercise 2.2: INCORRECT"
  
    
  DoG1_student = cv2.imread(os.path.expanduser("DoG1.png"), 0) / 255.0
  DoG1_correct = cv2.imread(os.path.expanduser("answer-images/DoG1.png"), 0) / 255.0
  if DoG1_student.shape == DoG1_correct.shape and np.all(np.abs(DoG1_student - DoG1_correct) < 0.01):
    print "Exercise 2.3: Difference of Gaussians (sigma=1.6 and sigma=1) CORRECT"
  else:
    print "Exercise 2.3: Difference of Gaussians (sigma=1.6 and sigma=1) INCORRECT"
  DoG2_student = cv2.imread(os.path.expanduser("DoG2.png"), 0) / 255.0
  DoG2_correct = cv2.imread(os.path.expanduser("answer-images/DoG2.png"), 0) / 255.0
  if DoG2_student.shape == DoG2_correct.shape and np.all(np.abs(DoG2_student - DoG2_correct) < 0.01):
    print "Exercise 2.3: Difference of Gaussians (sigma=2.56 and sigma=1.6) CORRECT"
  else:
    print "Exercise 2.3: Difference of Gaussians (sigma=2.56 and sigma=1.6) INCORRECT"
  Laplacian1_student = cv2.imread(os.path.expanduser("Laplacian1.png"), 0) / 255.0
  Laplacian1_correct = cv2.imread(os.path.expanduser("answer-images/Laplacian1.png"), 0) / 255.0
  if Laplacian1_student.shape == Laplacian1_correct.shape and np.all(np.abs(Laplacian1_student - Laplacian1_correct) < 0.01):
    print "Exercise 2.3: Laplacian (sigma=1) CORRECT"
  else:
    print "Exercise 2.3: Laplacian (sigma=1) INCORRECT"  
  Laplacian2_student = cv2.imread(os.path.expanduser("Laplacian2.png"), 0) / 255.0
  Laplacian2_correct = cv2.imread(os.path.expanduser("answer-images/Laplacian2.png"), 0) / 255.0
  if Laplacian2_student.shape == Laplacian2_correct.shape and np.all(np.abs(Laplacian2_student - Laplacian2_correct) < 0.01):
    print "Exercise 2.3: Laplacian (sigma=1.6) CORRECT"
  else:
    print "Exercise 2.3: Laplacian (sigma=1.6) INCORRECT"    
    
  
  import sys, select 
  print "Press enter or any key on one of the images to exit"
  while True:
    if cv2.waitKey(100) != -1:
      break
    # http://stackoverflow.com/questions/1335507/keyboard-input-with-timeout-in-python
    i, o, e = select.select( [sys.stdin], [], [], 0.1 )
    if i:
      break


