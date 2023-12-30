import numpy as np
import cv2

def polyfit2d(X, Y, Z, deg):
  """Fit a 3D polynomial of degree deg"""
  vander = np.polynomial.polynomial.polyvander2d(X, Y, deg)
  vander = vander.reshape((-1, vander.shape[-1]))
  c      = np.linalg.lstsq(vander, Z.reshape((vander.shape[0],)), rcond=-1)[0]
  return   c.reshape(deg + 1)

def polyval2d(X, Y, C):
  """Evaluate a 2D Polynomial with coefficients C"""
  output = np.zeros_like(X)
  for i in range(C.shape[0]):
    for j in range(C.shape[1]):
      output += C[i, j] * (X ** i) * (Y ** j)
  return output

def polyval2dExpanded(X, Y, C):
  """Evaluate a 2D Polynomial with coefficients C, but more verbose"""
  Cf = C.flatten()
  X2 = X * X; X3 = X2 * X
  Y2 = Y * Y; Y3 = Y2 * Y
  return (((Cf[ 0]     ) + (Cf[ 1]      * Y) + (Cf[ 2]      * Y2) + (Cf[ 3]      * Y3)) +
          ((Cf[ 4] * X ) + (Cf[ 5] * X  * Y) + (Cf[ 6] * X  * Y2) + (Cf[ 7] * X  * Y3)) +
          ((Cf[ 8] * X2) + (Cf[ 9] * X2 * Y) + (Cf[10] * X2 * Y2) + (Cf[11] * X2 * Y3)) +
          ((Cf[12] * X3) + (Cf[13] * X3 * Y) + (Cf[14] * X3 * Y2) + (Cf[15] * X3 * Y3)))

def createMask(data):
  """Find the largest connected non-zero region, and mask it out"""
  # Create the "Raw Mask" from the measurements
  mask = np.clip(data[..., 1] * data[..., 2], 0, 1)

  # Find the largest contour and extract it
  contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  maxContour = 0
  for contour in contours:
   contourSize = cv2.contourArea(contour)
   if contourSize > maxContour:
     maxContour     = contourSize
     maxContourData = contour
  
  # Create a mask from the largest contour
  mask = np.zeros_like(mask)
  mask = cv2.fillPoly(mask, [maxContourData], 1)
  return mask

def calcCoeffs(sideId, data, pixToRectFunc, polynomialDegree=3):
  """Fit a 2D polynomial using data, mapping from Screen UVs to either Camera Pixel or Rectilinear Coordinates"""
  # Prepare the mask of valid measurements
  mask = createMask(data)
  data *= mask[...,None]


  # Extract the Valid Measurements and arrange into a flat array
  coordinates          = np.zeros((data.shape[0], data.shape[1], 2), dtype=int)
  coordinates[:, :, 1] = np.arange(0, data.shape[0])[:, None]  # Prepare the coordinates Y
  coordinates[:, :, 0] = np.arange(0, data.shape[1])[None, :]  # Prepare the coordinates X

  # Sample the non-zero indices from the flattened coordinates and data arrays
  non_zero_indices     = np.nonzero(mask.reshape(-1))[0] # Get non-zero mask indices
  non_zero_data = data[..., 1:].reshape(-1, 2)[non_zero_indices].astype(float) / 255

  # Fit the polynomial to camera image pixels
  #non_zero_coordinates = coordinates .reshape(-1, 2)[non_zero_indices].astype(float) / widthData.shape[0]

  # Fit the polynomial to rectilinear coordinates
  non_zero_pixel_coordinates = coordinates.reshape(-1, 1, 2)[non_zero_indices].astype(float)
  non_zero_coordinates = pixToRectFunc(sideId, non_zero_pixel_coordinates).reshape(-1, 2)

  # Fit the multidimensional polynomials
  x_coeffs = polyfit2d(non_zero_data[:, 1],        # "Input"  X Axis
                       non_zero_data[:, 0],        # "Input"  Y Axis
                       non_zero_coordinates[:, 0], # "Output" X Axis
                       np.asarray([polynomialDegree,
                                   polynomialDegree]))

  y_coeffs = polyfit2d(non_zero_data[:, 1],        # "Input"  X Axis
                       non_zero_data[:, 0],        # "Input"  Y Axis
                       non_zero_coordinates[:, 1], # "Output" Y Axis
                       np.asarray([polynomialDegree,
                                   polynomialDegree]))

  return x_coeffs, y_coeffs

