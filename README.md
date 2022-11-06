# PNSCalibrationTools
The output of the *optical calibration* is only compatible with the [forked version of ProjectEsky-UnityIntegration](https://github.com/JurajVincur/ProjectEsky-UnityIntegration)

The output of the *between-sensor calibration* provides local unity pose of *sensor i* which is supposed to be the child of the *sensor 0*. Index of the sensor is given by the order provided in command line arguments.

## Dependencies
- Python packages: PySide6, opencv, pyrealsense2, numpy, scipy
- SDK libraries (copy only files for devices that you would like to use):
  - Ultraleap - copy Ultraleap\LeapSDK\lib\x64\LeapC.dll to PNSCalibrationTools\src\utils\libs\ultraleap
  - Xvisio - copy dlls from XSlamSDK-rc\lib to PNSCalibrationTools\src\utils\libs\xvisio
  - Antilatency - copy AntilatencySDK dlls to PNSCalibrationTools\src\utils\libs\antilatency
  
## Optical calibration
1. Open a command prompt and change the current working directory to PNSCalibrationTools\src
2. Run command *python v2Widget.py*
3. In the opened window select a device that you would like to use for the optical calibration (device needs to be already connected to the PC)
4. Change display index so the second black window would appear on PNS screens (PNS needs to be already connected to the PC)
5. Click on *Create mask* and wait for procedure to finish (a mask image should appear in the bottom area)
6. Click on *Measure width bits* and wait for procedure to finish (a gradient image should appear in the bottom area)
7. Click on *Measure height bits* and wait for procedure to finish (a new gradient image should appear in the bottom area)
8. Click on *Fit a 3D polynomial* and wait for procedure to finish (a charuco board should appear in the bottom area)
9. Copy calibration output (JSON part) from command prompt to the target file.

## Between-sensor calibration
1. Open a command prompt and change the current working directory to PNSCalibrationTools\src
2. Run command *python betweenSensor.py -s SENSORS* where SENSORS is a space separated list of sensors
3. Show a printed Aruco marker with the id 5 and a length of 10cm (other lengths can be configured via -l switch) to the sensors from different positions
4. Hit q
5. Copy printed positions and rotations to the Unity
